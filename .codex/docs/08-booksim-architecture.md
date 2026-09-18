# BookSim architecture study

BookSim is a separate checkout (`28f4329`, `master`) and has not yet been
modified or integrated. The next study will identify its `Network`, topology,
router/channel, packet/flit/VC, traffic-manager, routing, configuration, clock,
and statistics classes, then run a standalone smoke test if practical.

The desired adapter surface is conceptual only at this stage:
`create_network`, `inject_packet`, `advance_cycle`, `query_completion`, and
`collect_stats`. Whether those map cleanly to existing BookSim objects is an
open source question.
