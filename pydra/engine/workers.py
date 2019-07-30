import asyncio
import sys
import re
from tempfile import gettempdir

import concurrent.futures as cf

from .helpers import create_pyscript, read_and_display, save

import logging

logger = logging.getLogger("pydra.worker")


class Worker:
    def __init__(self, loop=None):
        logger.debug(f"Initializing {self.__class__.__name__}")
        self.loop = loop
        self._distributed = False

    def run_el(self, interface, **kwargs):
        raise NotImplementedError

    def close(self):
        pass

    async def fetch_finished(self, futures):
        """Awaits asyncio ``Tasks`` until one is finished

        Parameters
        ----------
        futures : set of ``Futures``
            Pending tasks

        Returns
        -------
        done : set
            Finished or cancelled tasks
        """
        done = set()
        try:
            done, pending = await asyncio.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )
        except ValueError:
            # nothing pending!
            pending = set()

        assert (
            done.union(pending) == futures
        ), "all tasks from futures should be either in done or pending"
        logger.debug(f"Tasks finished: {len(done)}")
        return pending


class DistributedWorker(Worker):
    """Base Worker for distributed execution"""

    def __init__(self, loop=None):
        super(DistributedWorker, self).__init__(loop)
        self._distributed = True

    def _prepare_runscripts(self, task, interpreter="/bin/sh"):
        script_dir = (
            task.cache_dir / f"{self.__class__.__name__}_scripts" / task.checksum
        )
        script_dir.mkdir(parents=True, exist_ok=True)
        if not (script_dir / "_task.pkl").exists():
            save(script_dir, task=task)
        pyscript = create_pyscript(script_dir, task.checksum)
        batchscript = script_dir / f"batchscript_{task.checksum}.sh"
        bcmd = "\n".join(
            (
                f"#!{interpreter}",
                f"#SBATCH --output={str(script_dir / 'slurm-%j.out')}",
                f"{sys.executable} {str(pyscript)}",
            )
        )
        with batchscript.open("wt") as fp:
            fp.writelines(bcmd)
        return script_dir, pyscript, batchscript

    @staticmethod
    async def _awaitable(jobid):
        """Readily available coroutine"""
        return jobid


class SerialPool:
    """ a simply class to imitate a pool like in cf"""

    def submit(self, interface, **kwargs):
        self.res = interface(**kwargs)

    def result(self):
        return self.res

    def done(self):
        return True


class SerialWorker(Worker):
    def __init__(self):
        logger.debug("Initialize SerialWorker")
        self.pool = SerialPool()

    def run_el(self, interface, **kwargs):
        self.pool.submit(interface=interface, **kwargs)
        return self.pool

    def close(self):
        pass


class ConcurrentFuturesWorker(Worker):
    def __init__(self, nr_proc=None):
        super(ConcurrentFuturesWorker, self).__init__()
        self.nr_proc = nr_proc
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.nr_proc)
        # self.loop = asyncio.get_event_loop()
        logger.debug("Initialize ConcurrentFuture")

    def run_el(self, runnable, **kwargs):
        assert self.loop, "No event loop available to submit tasks"
        task = asyncio.create_task(self.exec_as_coro(runnable))
        return task

    async def exec_as_coro(self, runnable):
        res = await self.loop.run_in_executor(self.pool, runnable._run)
        return res

    def close(self):
        self.pool.shutdown()


class SlurmWorker(DistributedWorker):
    _cmd = "sbatch"

    def __init__(self, poll_delay=1, sbatch_args=None, **kwargs):
        """Initialize Slurm Worker

        Parameters
        ----------
        poll_delay : seconds
            Delay between polls to slurmd
        sbatch_args : str
            Additional sbatch arguments
        """
        super(SlurmWorker, self).__init__()
        if not poll_delay or poll_delay < 0:
            poll_delay = 0
        self.poll_delay = poll_delay
        self.sbatch_args = sbatch_args or ""
        self.sacct_re = re.compile(
            "(?P<jobid>\\d*) +(?P<status>\\w*)\\+? +" "(?P<exit_code>\\d+):\\d+"
        )

    def run_el(self, runnable):
        """
        Worker submission API
        """
        script_dir, _, batch_script = self._prepare_runscripts(runnable)
        if (script_dir / script_dir.parts[1]) == gettempdir():
            logger.warning("Temporary directories may not be shared across computers")
        task = asyncio.create_task(self._submit_job(runnable, batch_script))
        return task

    async def _submit_job(self, task, batchscript):
        """Coroutine that submits task runscript and polls job until completion or error."""
        sargs = self.sbatch_args.split()
        jobname = re.search(r"(?<=-J )\S+|(?<=--job-name=)\S+", self.sbatch_args)
        if not jobname:
            jobname = ".".join((task.name, task.checksum))
            sargs.append(f"--job-name={jobname}")
        output = re.search(r"(?<=-o )\S+|(?<=--output=)\S+", self.sbatch_args)
        if not output:
            output = str(batchscript.parent / "slurm-%j.out")
            sargs.append(f"--output={output}")
        sargs.append(str(batchscript))
        # TO CONSIDER: add random sleep to avoid overloading calls
        _, stdout, _ = await read_and_display("sbatch", *sargs, hide_display=True)
        jobid = re.search(r"\d+", stdout)
        if not jobid:
            raise RuntimeError("Could not extract job ID")
        jobid = jobid.group()
        # intermittent polling
        while True:
            # 3 possibilities
            # False: job is still pending/working
            # True: job is complete
            # Exception: Polling / job failure
            done = await self._poll_job(jobid)
            if done:
                return True
            await asyncio.sleep(self.poll_delay)

    async def _poll_job(self, jobid):
        cmd = ("squeue", "-h", "-j", jobid)
        logger.debug(f"Polling job {jobid}")
        rc, stdout, stderr = await read_and_display(*cmd, hide_display=True)
        if not stdout or "slurm_load_jobs error" in stderr:
            # job is no longer running - check exit code
            status = await self._verify_exit_code(jobid)
            return status
        return False

    async def _verify_exit_code(self, jobid):
        cmd = ("sacct", "-n", "-X", "-j", jobid, "-o", "JobID,State,ExitCode")
        _, stdout, _ = await read_and_display(*cmd, hide_display=True)
        if not stdout:
            raise RuntimeError("Job information not found")
        m = self.sacct_re.search(stdout)
        if int(m.group("exit_code")) != 0 or m.group("status") != "COMPLETED":
            # TODO: potential for requeuing
            raise Exception("Job failed")
        return True
