import time
import pytest
import typing as _t

#  from mercury import ID, Network, Flow, Message, FlowMessage
#  from mercury.actors import SimpleSyncActor, Reflector
#  from mercury.resolvers import UnorderedResolver


#  @pytest.mark.parametrize('cl, left, right', [
#  (2, 0.25, 0.5),
#  (3, 0.675, 0.5),
#  (4, 0.675, 0.5625),
#  (5, 0.70625, 0.5625)
#  ])
#  def test_ordering(cl, left, right):
#  resolver = UnorderedResolver(chain_limit=cl)
#  n = Network(resolver, {
#  "A": SimpleSyncActor("A"),
#  "B": Reflector("B")
#  })
#  n.add_connection("A", "B")

#  flow = Flow(inv=1.0, cash=-10.0)

#  n.context_for("A").send("B", [FlowMessage(flow)])
#  n.resolve()
