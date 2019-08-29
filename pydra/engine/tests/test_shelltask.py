# -*- coding: utf-8 -*-

import typing as ty
import os, shutil
import pytest


from ..task import ShellCommandTask
from ..submitter import Submitter
from ..core import Workflow

if bool(shutil.which("sbatch")):
    Plugins = ["cf", "slurm"]
else:
    Plugins = ["cf"]


def test_shell_cmd_1_nosubm(tmpdir):
    """ simple command, no arguments
        no submitter
    """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly()
    assert res.output.stdout == str(shelly.output_dir) + "\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_1(plugin):
    """ simple command, no arguments
        using submitter
    """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == str(shelly.output_dir) + "\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_1_strip(plugin):
    """ simple command, no arguments
        strip option to remove \n at the end os stdout
    """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd, strip=True)
    assert shelly.cmdline == " ".join(cmd)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == str(shelly.output_dir)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_2_nosubm():
    """ a command with arguments, cmd and args given as executable
        no submitter
    """
    cmd = ["echo", "hail", "pydra"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_2(plugin):
    """ a command with arguments, cmd and args given as executable
        using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)
    res = shelly.result()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_2a_nosubm():
    """ a command with arguments, using executable and args
        no submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo " + " ".join(cmd_args)
    res = shelly()
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_2a(plugin):
    """ a command with arguments, using executable and args
        using submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo " + " ".join(cmd_args)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)
    res = shelly.result()
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_2b_nosubm():
    """ a command with arguments, using strings for executable and args
        no submitter
    """
    cmd_exec = "echo"
    cmd_args = "pydra"
    # separate command into exec + args
    shelly = ShellCommandTask(executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo pydra"
    res = shelly()
    assert res.output.stdout == "pydra\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_2b(plugin):
    """ a command with arguments, using  strings executable and args
        using submitter
    """
    cmd_exec = "echo"
    cmd_args = "pydra"
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo pydra"
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)
    res = shelly.result()
    assert res.output.stdout == "pydra\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


# tests with State


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_3(plugin):
    """ commands without arguments
        splitter = executable
    """
    cmd = ["pwd", "whoami"]

    # all args given as executable
    shelly = ShellCommandTask(name="shelly", executable=cmd).split("executable")
    # TODO: doesnt make sense for tasks with splitter
    #    assert shelly.cmdline == " ".join(cmd)
    res = shelly(plugin=plugin)
    assert res[0].output.stdout == f"{str(shelly.output_dir[0])}\n"
    if "USER" in os.environ:
        assert res[1].output.stdout == f"{os.environ['USER']}\n"
    else:
        assert res[1].output.stdout
    assert res[0].output.return_code == res[1].output.return_code == 0
    assert res[0].output.stderr == res[1].output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_4(plugin):
    """ a command with arguments, using executable and args
        splitter=args
    """
    cmd_exec = "echo"
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args).split(
        splitter="args"
    )
    assert shelly.inputs.executable == "echo"
    assert shelly.inputs.args == ["nipype", "pydra"]
    # this doesnt work, cmdline gives echo nipype pydra
    # assert shelly.cmdline == "echo pydra"
    res = shelly(plugin=plugin)

    assert res[0].output.stdout == "nipype\n"
    assert res[1].output.stdout == "pydra\n"

    assert res[0].output.return_code == res[1].output.return_code == 0
    assert res[0].output.stderr == res[1].output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_5(plugin):
    """ a command with arguments
        using splitter and combiner for args
    """
    cmd_exec = "echo"
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = (
        ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
        .split(splitter="args")
        .combine("args")
    )
    assert shelly.inputs.executable == "echo"
    assert shelly.inputs.args == ["nipype", "pydra"]
    # this doesnt work, cmdline gives echo nipype pydra
    # assert shelly.cmdline == "echo pydra"
    res = shelly(plugin=plugin)

    assert res[0][0].output.stdout == "nipype\n"
    assert res[0][1].output.stdout == "pydra\n"


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_6(plugin):
    """ a command with arguments,
        outer splitter for executable and args
    """
    cmd_exec = ["echo", ["echo", "-n"]]
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args).split(
        splitter=["executable", "args"]
    )
    assert shelly.inputs.executable == ["echo", ["echo", "-n"]]
    assert shelly.inputs.args == ["nipype", "pydra"]
    # this doesnt work, cmdline gives echo nipype pydra
    # assert shelly.cmdline == "echo pydra"
    res = shelly(plugin=plugin)

    assert res[0].output.stdout == "nipype\n"
    assert res[1].output.stdout == "pydra\n"
    assert res[2].output.stdout == "nipype"
    assert res[3].output.stdout == "pydra"

    assert (
        res[0].output.return_code
        == res[1].output.return_code
        == res[2].output.return_code
        == res[3].output.return_code
        == 0
    )
    assert (
        res[0].output.stderr
        == res[1].output.stderr
        == res[2].output.stderr
        == res[3].output.stderr
        == ""
    )


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_7(plugin):
    """ a command with arguments,
        outer splitter for executable and args, and combiner=args
    """
    cmd_exec = ["echo", ["echo", "-n"]]
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = (
        ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
        .split(splitter=["executable", "args"])
        .combine("args")
    )
    assert shelly.inputs.executable == ["echo", ["echo", "-n"]]
    assert shelly.inputs.args == ["nipype", "pydra"]
    # this doesnt work, cmdline gives echo nipype pydra
    # assert shelly.cmdline == "echo pydra"
    res = shelly(plugin=plugin)

    assert res[0][0].output.stdout == "nipype\n"
    assert res[0][1].output.stdout == "pydra\n"

    assert res[1][0].output.stdout == "nipype"
    assert res[1][1].output.stdout == "pydra"


# saving files


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_8(plugin):
    """ creating a file and using output_files_spec to save with Results
    """
    cmd = ["touch", "a.txt"]
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_files_spec={"file": "a.txt"}
    )
    assert shelly.cmdline == " ".join(cmd)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)
    res = shelly.result()
    assert res.output.stdout == ""
    assert res.output.return_code == 0
    assert res.output.stderr == ""
    assert res.output.file.name == "a.txt"


# tests with workflows


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_shell_cmd_1(plugin):
    """ a workflow with two connected commands"""
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2"])
    wf.inputs.cmd1 = "pwd"
    wf.inputs.cmd2 = "ls"
    wf.add(ShellCommandTask(name="shelly_pwd", executable=wf.lzin.cmd1, strip=True))
    wf.add(
        ShellCommandTask(
            name="shelly_ls", executable=wf.lzin.cmd2, args=wf.shelly_pwd.lzout.stdout
        )
    )

    wf.set_output([("out", wf.shelly_ls.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert "_result.pklz" in res.output.out
    assert "_task.pklz" in res.output.out
