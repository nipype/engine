from ..core import Workflow
from ..task import to_task
from ..submitter import Submitter

import time


@to_task
def sleep_add_one(x):
    time.sleep(1)
    return x + 1


def test_concurrent_wf():
    # concurrent workflow
    # A --> C
    # B --> D
    wf = Workflow("new_wf", input_spec=["x", "y"])
    wf.inputs.x = 5
    wf.inputs.y = 10
    wf.add(sleep_add_one(name="taska", x=wf.lzin.x))
    wf.add(sleep_add_one(name="taskb", x=wf.lzin.y))
    wf.add(sleep_add_one(name="taskc", x=wf.taska.lzout.out))
    wf.add(sleep_add_one(name="taskd", x=wf.taskb.lzout.out))
    wf.set_output([("out1", wf.taskc.lzout.out), ("out2", wf.taskd.lzout.out)])
    # wf.plugin = 'cf'
    # res = wf.run()
    with Submitter("cf") as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out1 == 7
    assert res.output.out2 == 12


def test_wf_in_wf():
    """WF(A --> SUBWF(A --> B) --> B)"""
    wf = Workflow(name="wf_in_wf", input_spec=["x"])
    wf.inputs.x = 3
    wf.add(sleep_add_one(name="wf_a", x=wf.lzin.x))

    # workflow task
    subwf = Workflow(name="sub_wf", input_spec=["x"])
    subwf.add(sleep_add_one(name="sub_a", x=subwf.lzin.x))
    subwf.add(sleep_add_one(name="sub_b", x=subwf.sub_a.lzout.out))
    subwf.set_output([("out", subwf.sub_b.lzout.out)])
    # connect, then add
    subwf.inputs.x = wf.wf_a.lzout.out
    wf.add(subwf)

    wf.add(sleep_add_one(name="wf_b", x=wf.sub_wf.lzout.out))
    wf.set_output([("out", wf.wf_b.lzout.out)])

    with Submitter("cf") as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 7


def test_wf2():
    """ workflow as a node
        workflow-node with one task and no splitter
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(sleep_add_one(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wfnd.inputs.x = 2

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])

    with Submitter("cf") as sub:
        sub(wf)

    res = wf.result()
    assert res.output.out == 3


def test_wf_with_state():
    wf = Workflow(name="wf_with_state", input_spec=["x"])
    wf.add(sleep_add_one(name="taska", x=wf.lzin.x))
    wf.add(sleep_add_one(name="taskb", x=wf.taska.lzout.out))

    wf.inputs.x = [1, 2, 3]
    wf.split("x")
    wf.set_output([("out", wf.taskb.lzout.out)])

    with Submitter("cf") as sub:
        sub(wf)

    res = wf.result()

    assert res[0].output.out == 3
    assert res[1].output.out == 4
    assert res[2].output.out == 5
