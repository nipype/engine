"""
Implement processing nodes.

.. admonition :: Notes:

    * Environment specs

        1. neurodocker json
        2. singularity file+hash
        3. docker hash
        4. conda env
        5. niceman config
        6. environment variables

    * Monitors/Audit

        1. internal monitor
        2. external monitor
        3. callbacks

    * Resuming

        1. internal tracking
        2. external tracking (DMTCP)

    * Provenance

        1. Local fragments
        2. Remote server

    * Isolation

        1. Working directory
        2. File (copy to local on write)
        3. read only file system

    * `Original implementation
      <https://colab.research.google.com/drive/1RRV1gHbGJs49qQB1q1d5tQEycVRtuhw6>`__

"""
import attr
import cloudpickle as cp
import inspect
import typing as ty
from pathlib import Path

from .core import TaskBase
from ..utils.messenger import AuditFlag
from .specs import (
    File,
    BaseSpec,
    SpecInfo,
    ShellSpec,
    ShellOutSpec,
    ContainerSpec,
    DockerSpec,
    SingularitySpec,
    attr_fields,
)
from .helpers import ensure_list, execute
from .helpers_file import template_update, is_local_file


class FunctionTask(TaskBase):
    """Wrap a Python callable as a task element."""

    def __init__(
        self,
        func: ty.Callable,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        name=None,
        output_spec: ty.Optional[BaseSpec] = None,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        func : :obj:`callable`
            A Python executable function.
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        cache_locations : :obj:`list` of :obj:`os.pathlike`
            List of alternative cache locations.
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        messenger_args :
            TODO
        messengers :
            TODO
        name : :obj:`str`
            Name of this task.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.

        """
        if input_spec is None:
            input_spec = SpecInfo(
                name="Inputs",
                fields=[
                    (
                        val.name,
                        attr.ib(
                            default=val.default,
                            type=val.annotation,
                            metadata={
                                "help_string": f"{val.name} parameter from {func.__name__}"
                            },
                        ),
                    )
                    if val.default is not inspect.Signature.empty
                    else (
                        val.name,
                        attr.ib(
                            type=val.annotation, metadata={"help_string": val.name}
                        ),
                    )
                    for val in inspect.signature(func).parameters.values()
                ]
                + [("_func", attr.ib(default=cp.dumps(func), type=str))],
                bases=(BaseSpec,),
            )
        else:
            input_spec.fields.append(
                ("_func", attr.ib(default=cp.dumps(func), type=str))
            )
        self.input_spec = input_spec
        if name is None:
            name = func.__name__
        super(FunctionTask, self).__init__(
            name,
            inputs=kwargs,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            cache_locations=cache_locations,
        )
        if output_spec is None:
            if "return" not in func.__annotations__:
                output_spec = SpecInfo(
                    name="Output", fields=[("out", ty.Any)], bases=(BaseSpec,)
                )
            else:
                return_info = func.__annotations__["return"]
                if hasattr(return_info, "__name__") and hasattr(
                    return_info, "__annotations__"
                ):
                    output_spec = SpecInfo(
                        name=return_info.__name__,
                        fields=list(return_info.__annotations__.items()),
                        bases=(BaseSpec,),
                    )
                # Objects like int, float, list, tuple, and dict do not have __name__ attribute.
                elif hasattr(return_info, "__annotations__"):
                    output_spec = SpecInfo(
                        name="Output",
                        fields=list(return_info.__annotations__.items()),
                        bases=(BaseSpec,),
                    )
                elif isinstance(return_info, dict):
                    output_spec = SpecInfo(
                        name="Output",
                        fields=list(return_info.items()),
                        bases=(BaseSpec,),
                    )
                else:
                    if not isinstance(return_info, tuple):
                        return_info = (return_info,)
                    output_spec = SpecInfo(
                        name="Output",
                        fields=[
                            ("out{}".format(n + 1), t)
                            for n, t in enumerate(return_info)
                        ],
                        bases=(BaseSpec,),
                    )
        elif "return" in func.__annotations__:
            raise NotImplementedError("Branch not implemented")
        self.output_spec = output_spec

    def _run_task(self):
        inputs = attr.asdict(self.inputs)
        del inputs["_func"]
        self.output_ = None
        output = cp.loads(self.inputs._func)(**inputs)
        if output is not None:
            output_names = [el[0] for el in self.output_spec.fields]
            self.output_ = {}
            if len(output_names) > 1:
                if len(output_names) == len(output):
                    self.output_ = dict(zip(output_names, output))
                else:
                    raise Exception(
                        f"expected {len(self.output_spec.fields)} elements, "
                        f"but {len(output)} were returned"
                    )
            else:  # if only one element in the fields, everything should be returned together
                self.output_[output_names[0]] = output


class ShellCommandTask(TaskBase):
    """Wrap a shell command as a task element."""

    def __init__(
        self,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        name=None,
        output_spec: ty.Optional[SpecInfo] = None,
        strip=False,
        container=None,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        messenger_args :
            TODO
        messengers :
            TODO
        name : :obj:`str`
            Name of this task.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO

        """
        if input_spec is None:
            input_spec = SpecInfo(name="Inputs", fields=[], bases=(ShellSpec,))
        self.input_spec = input_spec
        if output_spec is None:
            output_spec = SpecInfo(name="Output", fields=[], bases=(ShellOutSpec,))

        self.output_spec = output_spec

        super(ShellCommandTask, self).__init__(
            name=name,
            inputs=kwargs,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
        )
        self.strip = strip
        self.container = container
        if self.container:
            if self.container not in ["docker", "singularity"]:
                raise Exception(
                    f"container can be either docker or singularity, "
                    f"but {self.container} provided"
                )

    @property
    def command_args(self):
        """Get command line arguments."""
        pos_args = []  # list for (position, command arg)
        for f in attr_fields(self.inputs):
            if f.name == "executable":
                pos = 0  # executable should be the first el. of the command
            elif f.name == "args":
                pos = -1  # assuming that args is the last el. of the command
            # if inp has position than it should be treated as a part of the command
            # metadata["position"] is the position in the command
            elif "position" in f.metadata:
                pos = f.metadata["position"]
                if not isinstance(pos, int) or pos < 1:
                    raise Exception(
                        f"position should be an integer > 0, but {pos} given"
                    )
            else:
                continue
            cmd_add = []
            if "argstr" in f.metadata:
                cmd_add.append(f.metadata["argstr"])
            # if f.metadata.get("copyfile") in [True, False]:
            #    value = str(self.inputs.map_copyfiles[f.name])
            # else:
            value = getattr(self.inputs, f.name)
            if is_local_file(f):
                value = str(value)
            # changing path to the cpath (the directory should be mounted)
            if getattr(self, "bind_paths", None) and is_local_file(f):
                lpath = Path(value)
                cdir = self.bind_paths[lpath.parent][0]
                cpath = cdir.joinpath(lpath.name)
                value = str(cpath)
            if f.type is bool:
                if value is not True:
                    break
            else:
                cmd_add += ensure_list(value, tuple2list=True)
            if cmd_add is not None:
                pos_args.append((pos, cmd_add))
        # sorting all elements of the command
        pos_args.sort()
        # if args available, they should be moved at the of the list
        if pos_args[0][0] == -1:
            pos_args.append(pos_args.pop(0))
        # dropping the position index
        cmd_args = []
        for el in pos_args:
            cmd_args += el[1]
        return cmd_args

    @command_args.setter
    def command_args(self, args: ty.Dict):
        self.inputs = attr.evolve(self.inputs, **args)

    def container_args(self):
        """ there is no container args for the Shell Task if container is not provided"""
        if self.container == "docker":
            self.inputs.container = "docker"
            self.inputs.container_xargs = ["--rm"]
            self.output_cpath = Path("/output_pydra")
            cargs = DockerTask.container_args(self)
        elif self.container == "singularity":
            self.inputs.container = "singularity"
            self.output_cpath = Path("/output_pydra")
            cargs = SingularityTask.container_args(self)
        else:
            cargs = []
        return cargs

    @property
    def bind_paths(self):
        """Return bound mount points: ``dict(lpath: (cpath, mode))``."""
        bind_paths = {}
        output_dir_cpath = None
        if not getattr(self.inputs, "bindings", None):
            self.inputs.bindings = []
        for binding in self.inputs.bindings:
            if len(binding) == 3:
                lpath, cpath, mode = binding
            elif len(binding) == 2:
                lpath, cpath, mode = binding + ["rw"]
            else:
                raise Exception(
                    f"binding should have length 2, 3, or 4, it has {len(binding)}"
                )
            if Path(lpath) == self.output_dir:
                output_dir_cpath = cpath
            if mode is None:
                mode = "rw"  # default
            bind_paths[Path(lpath)] = (Path(cpath), mode)
        # output_dir is added to the bindings if not part of self.inputs.bindings
        if not output_dir_cpath:
            bind_paths[self.output_dir] = (self.output_cpath, "rw")
        return bind_paths

    def binds(self, opt):
        """
        Specify mounts to bind from local filesystems to container and working directory.

        Uses py:meth:`binds_paths`

        """
        bargs = []
        for (key, val) in self.bind_paths.items():
            bargs.extend([opt, "{0}:{1}:{2}".format(key, val[0], val[1])])
        return bargs

    @property
    def cmdline(self):
        """Get the actual command line that will be submitted."""
        orig_inputs = attr.asdict(self.inputs)
        modified_inputs = template_update(self.inputs)
        if modified_inputs is not None:
            self.inputs = attr.evolve(self.inputs, **modified_inputs)
        cmdline = " ".join(self.container_args() + self.command_args)
        self.inputs = attr.evolve(self.inputs, **orig_inputs)
        return cmdline

    def _run_task(self):
        self.output_ = None
        args = self.container_args() + self.command_args
        if args:
            # removing emty strings
            args = [el for el in args if el not in ["", " "]]
            keys = ["return_code", "stdout", "stderr"]
            values = execute(args, strip=self.strip)
            self.output_ = dict(zip(keys, values))
            if self.output_["return_code"]:
                raise RuntimeError(self.output_["stderr"])


class ContainerTask(ShellCommandTask):
    """Extend shell command task for containerized execution."""

    def __init__(
        self,
        name,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        output_cpath="/output_pydra",
        output_spec: ty.Optional[SpecInfo] = None,
        strip=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        name : :obj:`str`
            Name of this task.
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        messenger_args :
            TODO
        messengers :
            TODO
        output_cpath : :obj:`str`
            Output path within the container filesystem.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO

        """
        if input_spec is None:
            input_spec = SpecInfo(name="Inputs", fields=[], bases=(ContainerSpec,))
        self.output_cpath = Path(output_cpath)
        super(ContainerTask, self).__init__(
            name=name,
            input_spec=input_spec,
            output_spec=output_spec,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            strip=strip,
            **kwargs,
        )

        if self.inputs.container is None:
            raise AttributeError("Container software is not specified")
        if self.inputs.image is None:
            raise AttributeError("Container image is not specified")


class DockerTask(ContainerTask):
    """Extend shell command task for containerized execution with the Docker Engine."""

    def __init__(
        self,
        name,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        output_cpath="/output_pydra",
        output_spec: ty.Optional[SpecInfo] = None,
        strip=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        name : :obj:`str`
            Name of this task.
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        messenger_args :
            TODO
        messengers :
            TODO
        output_cpath : :obj:`str`
            Output path within the container filesystem.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO

        """
        if input_spec is None:
            input_spec = SpecInfo(name="Inputs", fields=[], bases=(DockerSpec,))

        super(DockerTask, self).__init__(
            name=name,
            input_spec=input_spec,
            output_spec=output_spec,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            strip=strip,
            output_cpath=output_cpath,
            **kwargs,
        )
        self.inputs.container_xargs = ["--rm"]

    def container_args(self):
        """Get container-specific CLI arguments."""
        assert self.inputs.container == "docker"
        cargs = ["docker", "run"]
        if self.inputs.container_xargs is not None:
            cargs.extend(self.inputs.container_xargs)

        cargs.extend(self.binds("-v"))
        cargs.extend(["-w", str(self.output_cpath)])
        cargs.append(self.inputs.image)

        return cargs


class SingularityTask(ContainerTask):
    """Extend shell command task for containerized execution with Singularity."""

    def __init__(
        self,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        output_spec: ty.Optional[SpecInfo] = None,
        strip=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        name : :obj:`str`
            Name of this task.
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        messenger_args :
            TODO
        messengers :
            TODO
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO

        """
        if input_spec is None:
            input_spec = SpecInfo(name="Inputs", fields=[], bases=(SingularitySpec,))
        super(SingularityTask, self).__init__(
            input_spec=input_spec,
            output_spec=output_spec,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            strip=strip,
            **kwargs,
        )

    def container_args(self):
        """Get container-specific CLI arguments."""
        assert self.inputs.container == "singularity"
        cargs = ["singularity", "exec"]
        if self.inputs.container_xargs is not None:
            cargs.extend(self.inputs.container_xargs)
        cargs.append(self.inputs.image)

        # insert bindings before image
        idx = len(cargs) - 1
        cargs[idx:-1] = self.binds("-B")
        return cargs
