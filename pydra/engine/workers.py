"""Execution workers."""
import asyncio
import sys, os, json
import re
from tempfile import gettempdir
from pathlib import Path
from shutil import copyfile, which

import concurrent.futures as cf

from .core import TaskBase
from .helpers import get_available_cpus, read_and_display_async, save, load_and_run

import logging

import random

logger = logging.getLogger("pydra.worker")


class Worker:
    """A base class for execution of tasks."""

    def __init__(self, loop=None):
        """Initialize the worker."""
        logger.debug(f"Initializing {self.__class__.__name__}")
        self.loop = loop

    def run_el(self, interface, **kwargs):
        """Return coroutine for task execution."""
        raise NotImplementedError

    def close(self):
        """Close this worker."""

    async def fetch_finished(self, futures):
        """
        Awaits asyncio's :class:`asyncio.Task` until one is finished.

        Parameters
        ----------
        futures : set of asyncio awaitables
            Task execution coroutines or asyncio :class:`asyncio.Task`

        Returns
        -------
        pending : set
            Pending asyncio :class:`asyncio.Task`.

        """
        done = set()
        try:
            done, pending = await asyncio.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )
        except ValueError:
            # nothing pending!
            pending = set()
        logger.debug(f"Tasks finished: {len(done)}")
        return pending


class DistributedWorker(Worker):
    """Base Worker for distributed execution."""

    def __init__(self, loop=None, max_jobs=None):
        """Initialize the worker."""
        super().__init__(loop=loop)
        self.max_jobs = max_jobs
        """Maximum number of concurrently running jobs."""
        self._jobs = 0

    async def fetch_finished(self, futures):
        """
        Awaits asyncio's :class:`asyncio.Task` until one is finished.

        Limits number of submissions based on
        py:attr:`DistributedWorker.max_jobs`.

        Parameters
        ----------
        futures : set of asyncio awaitables
            Task execution coroutines or asyncio :class:`asyncio.Task`

        Returns
        -------
        pending : set
            Pending asyncio :class:`asyncio.Task`.

        """
        done, unqueued = set(), set()
        job_slots = self.max_jobs - self._jobs if self.max_jobs else float("inf")
        if len(futures) > job_slots:
            # convert to list to simplify indexing
            logger.warning(f"Reducing queued jobs due to max jobs ({self.max_jobs})")
            futures = list(futures)
            futures, unqueued = set(futures[:job_slots]), set(futures[job_slots:])
        try:
            self._jobs += len(futures)
            done, pending = await asyncio.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )
        except ValueError:
            # nothing pending!
            pending = set()
        self._jobs -= len(done)
        logger.debug(f"Tasks finished: {len(done)}")
        # ensure pending + unqueued tasks persist
        return pending.union(unqueued)


class SerialPool:
    """A simple class to imitate a pool executor of concurrent futures."""

    def submit(self, interface, **kwargs):
        """Send new task."""
        self.res = interface(**kwargs)

    def result(self):
        """Get the result of a task."""
        return self.res

    def done(self):
        """Return whether the task is finished."""
        return True


class SerialWorker(Worker):
    """A worker to execute linearly."""

    def __init__(self):
        """Initialize worker."""
        logger.debug("Initialize SerialWorker")
        self.pool = SerialPool()

    def run_el(self, interface, rerun=False, **kwargs):
        """Run a task."""
        self.pool.submit(interface=interface, rerun=rerun, **kwargs)
        return self.pool

    def close(self):
        """Return whether the task is finished."""


class ConcurrentFuturesWorker(Worker):
    """A worker to execute in parallel using Python's concurrent futures."""

    def __init__(self, n_procs=None):
        """Initialize Worker."""
        super().__init__()
        self.n_procs = get_available_cpus() if n_procs is None else n_procs
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.n_procs)
        # self.loop = asyncio.get_event_loop()
        logger.debug("Initialize ConcurrentFuture")

    def run_el(self, runnable, rerun=False, **kwargs):
        """Run a task."""
        assert self.loop, "No event loop available to submit tasks"
        return self.exec_as_coro(runnable, rerun=rerun)

    async def exec_as_coro(self, runnable, rerun=False):
        """Run a task (coroutine wrapper)."""
        if isinstance(runnable, TaskBase):
            res = await self.loop.run_in_executor(self.pool, runnable._run, rerun)
        else:  # it could be tuple that includes pickle files with tasks and inputs
            ind, task_main_pkl, task_orig = runnable
            res = await self.loop.run_in_executor(
                self.pool, load_and_run, task_main_pkl, ind, rerun
            )
        return res

    def close(self):
        """Finalize the internal pool of tasks."""
        self.pool.shutdown()


class SlurmWorker(DistributedWorker):
    """A worker to execute tasks on SLURM systems."""

    _cmd = "sbatch"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
    )

    def __init__(self, loop=None, max_jobs=None, poll_delay=1, sbatch_args=None):
        """
        Initialize SLURM Worker.

        Parameters
        ----------
        poll_delay : seconds
            Delay between polls to slurmd
        sbatch_args : str
            Additional sbatch arguments
        max_jobs : int
            Maximum number of submitted jobs

        """
        super().__init__(loop=loop, max_jobs=max_jobs)
        if not poll_delay or poll_delay < 0:
            poll_delay = 0
        self.poll_delay = poll_delay
        self.sbatch_args = sbatch_args or ""
        self.error = {}

    def run_el(self, runnable, rerun=False):
        """Worker submission API."""
        script_dir, batch_script = self._prepare_runscripts(runnable, rerun=rerun)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        if isinstance(runnable, TaskBase):
            cache_dir = runnable.cache_dir
            name = runnable.name
            uid = runnable.uid
        else:  # runnable is a tuple (ind, pkl file, task)
            cache_dir = runnable[-1].cache_dir
            name = runnable[-1].name
            uid = f"{runnable[-1].uid}_{runnable[0]}"

        return self._submit_job(batch_script, name=name, uid=uid, cache_dir=cache_dir)

    def _prepare_runscripts(self, task, interpreter="/bin/sh", rerun=False):
        if isinstance(task, TaskBase):
            cache_dir = task.cache_dir
            ind = None
            uid = task.uid
        else:
            ind = task[0]
            cache_dir = task[-1].cache_dir
            uid = f"{task[-1].uid}_{ind}"

        script_dir = cache_dir / f"{self.__class__.__name__}_scripts" / uid
        script_dir.mkdir(parents=True, exist_ok=True)
        if ind is None:
            if not (script_dir / "_task.pkl").exists():
                save(script_dir, task=task)
        else:
            copyfile(task[1], script_dir / "_task.pklz")

        task_pkl = script_dir / "_task.pklz"
        if not task_pkl.exists() or not task_pkl.stat().st_size:
            raise Exception("Missing or empty task!")

        batchscript = script_dir / f"batchscript_{uid}.sh"
        python_string = f"""'from pydra.engine.helpers import load_and_run; load_and_run(task_pkl="{str(task_pkl)}", ind={ind}, rerun={rerun}) '
        """
        bcmd = "\n".join(
            (
                f"#!{interpreter}",
                f"#SBATCH --output={str(script_dir / 'slurm-%j.out')}",
                f"{sys.executable} -c " + python_string,
            )
        )
        with batchscript.open("wt") as fp:
            fp.writelines(bcmd)
        return script_dir, batchscript

    async def _submit_job(self, batchscript, name, uid, cache_dir):
        """Coroutine that submits task runscript and polls job until completion or error."""
        script_dir = cache_dir / f"{self.__class__.__name__}_scripts" / uid
        sargs = self.sbatch_args.split()
        jobname = re.search(r"(?<=-J )\S+|(?<=--job-name=)\S+", self.sbatch_args)
        if not jobname:
            jobname = ".".join((name, uid))
            sargs.append(f"--job-name={jobname}")
        output = re.search(r"(?<=-o )\S+|(?<=--output=)\S+", self.sbatch_args)
        if not output:
            output_file = str(script_dir / "slurm-%j.out")
        error = re.search(r"(?<=-e )\S+|(?<=--error=)\S+", self.sbatch_args)
        if not error:
            error_file = str(script_dir / "slurm-%j.err")
        else:
            error_file = None
        sargs.append(str(batchscript))
        # TO CONSIDER: add random sleep to avoid overloading calls
        rc, stdout, stderr = await read_and_display_async(
            "sbatch", *sargs, hide_display=True
        )
        jobid = re.search(r"\d+", stdout)
        if rc:
            raise RuntimeError(f"Error returned from sbatch: {stderr}")
        elif not jobid:
            raise RuntimeError("Could not extract job ID")
        jobid = jobid.group()
        if error_file:
            error_file = error_file.replace("%j", jobid)
        self.error[jobid] = error_file.replace("%j", jobid)
        # intermittent polling
        while True:
            # 3 possibilities
            # False: job is still pending/working
            # True: job is complete
            # Exception: Polling / job failure
            done = await self._poll_job(jobid)
            if done:
                if (
                    done in ["CANCELLED", "TIMEOUT", "PREEMPTED"]
                    and "--no-requeue" not in self.sbatch_args
                ):
                    # loading info about task with a specific uid
                    info_file = cache_dir / f"{uid}_info.json"
                    if info_file.exists():
                        checksum = json.loads(info_file.read_text())["checksum"]
                        if (cache_dir / f"{checksum}.lock").exists():
                            # for pyt3.8 we could you missing_ok=True
                            (cache_dir / f"{checksum}.lock").unlink()
                    cmd_re = ("scontrol", "requeue", jobid)
                    await read_and_display_async(*cmd_re, hide_display=True)
                else:
                    return True
            await asyncio.sleep(self.poll_delay)

    async def _poll_job(self, jobid):
        cmd = ("squeue", "-h", "-j", jobid)
        logger.debug(f"Polling job {jobid}")
        rc, stdout, stderr = await read_and_display_async(*cmd, hide_display=True)
        if not stdout or "slurm_load_jobs error" in stderr:
            # job is no longer running - check exit code
            status = await self._verify_exit_code(jobid)
            return status
        return False

    async def _verify_exit_code(self, jobid):
        cmd = ("sacct", "-n", "-X", "-j", jobid, "-o", "JobID,State,ExitCode")
        _, stdout, _ = await read_and_display_async(*cmd, hide_display=True)
        if not stdout:
            raise RuntimeError("Job information not found")
        m = self._sacct_re.search(stdout)
        error_file = self.error[jobid]
        if int(m.group("exit_code")) != 0 or m.group("status") != "COMPLETED":
            if m.group("status") in ["CANCELLED", "TIMEOUT", "PREEMPTED"]:
                return m.group("status")
            elif m.group("status") in ["RUNNING", "PENDING"]:
                return False
            # TODO: potential for requeuing
            # parsing the error message
            error_line = Path(error_file).read_text().split("\n")[-2]
            if "Exception" in error_line:
                error_message = error_line.replace("Exception: ", "")
            elif "Error" in error_line:
                error_message = error_line.replace("Exception: ", "")
            else:
                error_message = "Job failed (unknown reason - TODO)"
            raise Exception(error_message)
        return True


class SGEWorker(DistributedWorker):
    """A worker to execute tasks on SLURM systems."""

    _cmd = "qsub"
    _sacct_re = re.compile(
        "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
    )

    def __init__(
        self,
        loop=None,
        max_jobs=None,
        poll_delay=1,
        qsub_args=None,
        write_output_files=True,
        max_job_array_length=50,
        indirect_submit_host=None,
        max_threads=None,
        poll_for_result_file=False,
        default_threads_per_task=1,
        polls_before_checking_evicted=60,
    ):
        print("in __init__")
        """
        Initialize SLURM Worker.

        Parameters
        ----------
        poll_delay : seconds
            Delay between polls to slurmd
        qsub_args : str
            Additional qsub arguments
        max_jobs : int
            Maximum number of submitted jobs

        """
        super().__init__(loop=loop, max_jobs=max_jobs)
        if not poll_delay or poll_delay < 0:
            poll_delay = 0
        self.poll_delay = poll_delay
        self.qsub_args = qsub_args or ""
        self.error = {}
        self.write_output_files = (
            write_output_files  # set to False to avoid OSError: Too many open files
        )
        self.tasks_to_run_by_threads_requested = {}
        self.output_by_jobid = {}
        self.jobid_by_task_uid = {}
        self.max_job_array_length = max_job_array_length
        self.threads_used = 0
        self.job_completed_by_jobid = {}
        self.indirect_submit_host = indirect_submit_host
        self.max_threads = max_threads
        self.default_threads_per_task = default_threads_per_task
        self.poll_for_result_file = poll_for_result_file
        self.polls_before_checking_evicted = polls_before_checking_evicted
        print("end __init__")

    def run_el(self, runnable, rerun=False):
        """Worker submission API."""
        print("in run_el")
        (
            script_dir,
            batch_script,
            task_pkl,
            ind,
            threads_requested,
            output_dir,
        ) = self._prepare_runscripts(runnable, rerun=rerun)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        if isinstance(runnable, TaskBase):
            cache_dir = runnable.cache_dir
            name = runnable.name
            uid = runnable.uid
        else:  # runnable is a tuple (ind, pkl file, task)
            cache_dir = runnable[-1].cache_dir
            name = runnable[-1].name
            uid = f"{runnable[-1].uid}_{runnable[0]}"
        print("end run_el")

        return self._submit_job(
            batch_script,
            name=name,
            uid=uid,
            cache_dir=cache_dir,
            task_pkl=task_pkl,
            ind=ind,
            threads_requested=threads_requested,
            output_dir=output_dir,
        )

    def _prepare_runscripts(self, task, interpreter="/bin/sh", rerun=False):
        print("in _prepare_runscripts")
        print(f"task: {task}")
        print(f"task.cache_dir: {task.cache_dir}")
        print(f"task.cache_locations: {task.cache_locations}")
        print(f"task.output_dir: {task.output_dir}")
        if isinstance(task, TaskBase):
            cache_dir = task.cache_dir
            ind = None
            uid = task.uid
        else:
            ind = task[0]
            cache_dir = task[-1].cache_dir
            uid = f"{task[-1].uid}_{ind}"

        script_dir = cache_dir / f"{self.__class__.__name__}_scripts" / uid
        script_dir.mkdir(parents=True, exist_ok=True)
        if ind is None:
            if not (script_dir / "_task.pkl").exists():
                save(script_dir, task=task)
        else:
            copyfile(task[1], script_dir / "_task.pklz")

        task_pkl = script_dir / "_task.pklz"
        if not task_pkl.exists() or not task_pkl.stat().st_size:
            raise Exception("Missing or empty task!")

        batchscript = script_dir / f"batchscript_{uid}.job"

        # Set the threads_requested if given in the task input_spec - otherwise set to the default
        try:
            threads_requested = task[-1].inputs.sgeThreads
        except:
            try:
                threads_requested = task.inputs.sgeThreads
            except:
                try:
                    # itk variable for threads is num_threads - check if that has been set
                    threads_requested = task[-1].inputs.num_threads
                except:
                    try:
                        threads_requested = task.inputs.num_threads
                    except:
                        threads_requested = self.default_threads_per_task
        if not isinstance(threads_requested, int):
            threads_requested = self.default_threads_per_task

        if threads_requested not in self.tasks_to_run_by_threads_requested:
            self.tasks_to_run_by_threads_requested[threads_requested] = []
        self.tasks_to_run_by_threads_requested[threads_requested].append(
            (str(task_pkl), ind, rerun)
        )

        print("end _prepare_runscripts")
        return (
            script_dir,
            batchscript,
            task_pkl,
            ind,
            threads_requested,
            task.output_dir,
        )

    async def get_tasks_to_run(self, threads_requested):
        print("in get_tasks_to_run")
        # Extract the first N tasks to run
        tasks_to_run_copy, self.tasks_to_run_by_threads_requested[threads_requested] = (
            self.tasks_to_run_by_threads_requested[threads_requested][
                : self.max_job_array_length
            ],
            self.tasks_to_run_by_threads_requested[threads_requested][
                self.max_job_array_length :
            ],
        )
        print("end get_tasks_to_run")
        return tasks_to_run_copy

    async def _submit_jobs(
        self,
        batchscript,
        name,
        uid,
        cache_dir,
        threads_requested,
        output_dir,
        interpreter="/bin/sh",
    ):
        print("in _submit_jobs")
        # for threads_requested in self.tasks_to_run_by_threads_requested:
        if (
            len(self.tasks_to_run_by_threads_requested.get(threads_requested))
            <= self.max_job_array_length
        ):
            await asyncio.sleep(self.poll_delay)
        tasks_to_run = await self.get_tasks_to_run(threads_requested)

        if len(tasks_to_run) > 0:
            if self.max_threads is not None:
                while self.threads_used > self.max_threads - threads_requested * len(
                    tasks_to_run
                ):
                    await asyncio.sleep(self.poll_delay)
            self.threads_used += threads_requested * len(tasks_to_run)

            python_string = f"""import sys; from pydra.engine.helpers import load_and_run; task_pkls={[task_tuple for task_tuple in tasks_to_run]}; task_index=int(sys.argv[1])-1; load_and_run(task_pkl=task_pkls[task_index][0], ind=task_pkls[task_index][1], rerun=task_pkls[task_index][2])"""
            bcmd_job = "\n".join(
                (
                    f"#!{interpreter}",
                    f"{sys.executable} {Path(batchscript).with_suffix('.py')}"
                    + " $SGE_TASK_ID",
                )
            )

            bcmd_py = python_string

            with Path(batchscript).with_suffix(".py").open("wt") as fp:
                fp.write(bcmd_py)

            with batchscript.open("wt") as fp:
                fp.writelines(bcmd_job)

            # python_string = f"""\"import sys; from pydra.engine.helpers import load_and_run; task_pkls={[task_tuple for task_tuple in tasks_to_run]}; task_index=int(sys.argv[1])-1; load_and_run(task_pkl=task_pkls[task_index][0], ind=task_pkls[task_index][1], rerun=task_pkls[task_index][2])\""""
            # bcmd = "\n".join(
            #     (
            #         f"#!{interpreter}",
            #         f"{sys.executable} -c " + python_string + " $SGE_TASK_ID",
            #     )
            # )

            # with batchscript.open("wt") as fp:
            #     fp.writelines(bcmd)

            script_dir = cache_dir / f"{self.__class__.__name__}_scripts" / uid
            script_dir.mkdir(parents=True, exist_ok=True)
            sargs = ["-t"]
            sargs.append(f"1-{len(tasks_to_run)}")
            sargs = sargs + self.qsub_args.split()

            jobname = re.search(r"(?<=-N )\S+", self.qsub_args)

            if not jobname:
                jobname = ".".join((name, uid))
                sargs.append("-N")
                sargs.append(jobname)
            output = re.search(r"(?<=-o )\S+", self.qsub_args)
            sargs.append("-pe")
            sargs.append("smp")
            sargs.append(f"{threads_requested}")
            if not output:
                output_file = str(script_dir / "sge-%j.out")
                if self.write_output_files:
                    sargs.append("-o")
                    sargs.append(output_file)
            error = re.search(r"(?<=-e )\S+", self.qsub_args)
            if not error:
                error_file = str(script_dir / "sge-%j.out")
                if self.write_output_files:
                    sargs.append("-e")
                    sargs.append(error_file)
            else:
                error_file = None
            sargs.append(str(batchscript))

            await asyncio.sleep(random.uniform(0, 5))

            jobid = await self.submit_array_job(sargs, tasks_to_run, error_file)
            print(jobid)

            poll_counter = 0

            while True:
                # 3 possibilities
                # False: job is still pending/working
                # True: job is complete
                # Exception: Polling / job failure
                # done = await self._poll_job(jobid)
                if not self.poll_for_result_file:
                    result_file = output_dir / "_result.pklz"
                    if result_file.exists():
                        return True
                    elif poll_counter == self.polls_before_checking_evicted:
                        exit_status = await self._verify_exit_code(jobid)
                        if exit_status == "ERRORED":
                            jobid = await self._rerun_job_array(
                                cache_dir, uid, sargs, tasks_to_run, error_file
                            )
                        poll_counter = 0
                    poll_counter += 1
                    await asyncio.sleep(self.poll_delay)
                else:
                    done = await self._poll_job(jobid, cache_dir)
                    if done:
                        if done in ["ERRORED"]:  # If the SGE job was evicted, rerun it
                            jobid = await self._rerun_job_array(
                                cache_dir, uid, sargs, tasks_to_run, error_file
                            )
                        else:
                            # return True
                            self.job_completed_by_jobid[jobid] = True
                            self.threads_used -= threads_requested * len(tasks_to_run)
                            return True

                    # Don't poll exactly on the same interval to avoid overloading SGE
                    await asyncio.sleep(
                        random.uniform(max(0, self.poll_delay - 2), self.poll_delay + 2)
                    )

        print("end _submit_jobs")

    async def _rerun_job_array(self, cache_dir, uid, sargs, tasks_to_run, error_file):
        print("Rerunning job array")
        # # loading info about task with a specific uid
        info_file = cache_dir / f"{uid}_info.json"
        if info_file.exists():
            checksum = json.loads(info_file.read_text())["checksum"]
            if (cache_dir / f"{checksum}.lock").exists():
                # for pyt3.8 we could use missing_ok=True
                (cache_dir / f"{checksum}.lock").unlink()
        # If the previous job array failed, run the array's script again and get the new jobid
        jobid = await self.submit_array_job(sargs, tasks_to_run, error_file)
        return jobid

    async def submit_array_job(self, sargs, tasks_to_run, error_file):
        print("in submit_array_job")
        if self.indirect_submit_host is not None:
            indirect_submit_host_prefix = []
            indirect_submit_host_prefix.append("ssh")
            indirect_submit_host_prefix.append(self.indirect_submit_host)
            indirect_submit_host_prefix.append('""export SGE_ROOT=/opt/sge;')
            rc, stdout, stderr = await read_and_display_async(
                *indirect_submit_host_prefix,
                str(Path(which("qsub")).parent / "qsub"),
                *sargs,
                '""',
                hide_display=True,
            )
        else:
            rc, stdout, stderr = await read_and_display_async(
                "qsub", *sargs, hide_display=True
            )
        jobid = re.search(r"\d+", stdout)
        if rc:
            raise RuntimeError(f"Error returned from qsub: {stderr}")
        elif not jobid:
            raise RuntimeError("Could not extract job ID")
        jobid = jobid.group()
        self.output_by_jobid[jobid] = (rc, stdout, stderr)

        for task_pkl, ind, rerun in tasks_to_run:

            self.jobid_by_task_uid[Path(task_pkl).parent.name] = jobid

        if error_file:
            error_file = str(error_file).replace("%j", jobid)
        self.error[jobid] = str(error_file).replace("%j", jobid)
        print("end __init__")
        return jobid

    async def get_output_by_task_pkl(self, task_pkl):
        print("in get_output_by_task_pkl")
        jobid = self.jobid_by_task_uid.get(task_pkl.parent.name)
        while jobid == None:
            jobid = self.jobid_by_task_uid.get(task_pkl.parent.name)
            await asyncio.sleep(1)
        job_output = self.output_by_jobid.get(jobid)
        while job_output == None:
            job_output = self.output_by_jobid.get(jobid)
            await asyncio.sleep(1)
        print("end submit_array_job")
        return job_output

    async def _submit_job(
        self,
        batchscript,
        name,
        uid,
        cache_dir,
        task_pkl,
        ind,
        threads_requested,
        output_dir,
    ):
        print("in _submit_job")
        """Coroutine that submits task runscript and polls job until completion or error."""

        await self._submit_jobs(
            batchscript, name, uid, cache_dir, threads_requested, output_dir
        )
        jobname = ".".join((name, uid))

        if self.poll_for_result_file:
            return True
            # while True:
            # print(f"batscript={batchscript}")
            # print(f"name={name}")
            # print(f"uid={uid}")
            # print(f"cache_dir={cache_dir}")
            # print(f"task_pkl={task_pkl}")
            # print(f"ind={ind}")
            # print(f"threads_requested={threads_requested}")
            # result_file = output_dir / "_result.pklz"
            # if result_file.exists():
            #     return True
            # else:
            #     await asyncio.sleep(self.poll_delay)
        else:
            rc, stdout, stderr = await self.get_output_by_task_pkl(task_pkl)
            while True:
                jobid = self.jobid_by_task_uid.get(task_pkl.parent.name)
                if self.job_completed_by_jobid.get(jobid) == True:
                    return True
                else:
                    await asyncio.sleep(self.poll_delay)
        print("end __init__")

    async def _poll_job(self, jobid, cache_dir):
        print("in _poll_job")
        cmd = (f"qstat", "-j", jobid)
        logger.debug(f"Polling job {jobid}")
        rc, stdout, stderr = await read_and_display_async(*cmd, hide_display=True)

        if not stdout:
            # job is no longer running - check exit code
            status = await self._verify_exit_code(jobid)
            return status
        print("end _submit_job")
        return False

    async def _verify_exit_code(self, jobid):
        print("in _verify_exit_code")
        cmd = (f"qacct", "-j", jobid)
        rc, stdout, stderr = await read_and_display_async(*cmd, hide_display=True)

        # job is still pending/working
        if re.match(r"error: job id .* not found", stderr):
            return False

        # Read the qacct stdout into dictionary stdout_dict
        stdout_dict = {}
        for line in stdout.splitlines():
            key_value = line.split(None, 1)
            if len(key_value) > 1:
                stdout_dict[key_value[0]] = key_value[1]
            else:
                stdout_dict[key_value[0]] = None
        if not stdout:
            raise RuntimeError("Job information not found")
        m = self._sacct_re.search(stdout)
        error_file = self.error[jobid]
        if [int(s) for s in stdout_dict["failed"].split() if s.isdigit()][0] == 0:
            return True
        else:
            return "ERRORED"  # SGE job failed
        print("end _verify_exit_code")
        return True


class DaskWorker(Worker):
    """A worker to execute in parallel using Dask.distributed.
    This is an experimental implementation with limited testing.
    """

    def __init__(self, **kwargs):
        """Initialize Worker."""
        super().__init__()
        try:
            from dask.distributed import Client
        except ImportError:
            logger.critical("Please instiall Dask distributed.")
            raise
        self.client = None
        self.client_args = kwargs
        logger.debug("Initialize Dask")

    def run_el(self, runnable, rerun=False, **kwargs):
        """Run a task."""
        return self.exec_dask(runnable, rerun=rerun)

    async def exec_dask(self, runnable, rerun=False):
        """Run a task (coroutine wrapper)."""
        if self.client is None:
            from dask.distributed import Client

            self.client = await Client(**self.client_args, asynchronous=True)
        future = self.client.submit(runnable._run, rerun)
        result = await future
        return result

    def close(self):
        """Finalize the internal pool of tasks."""
        pass
