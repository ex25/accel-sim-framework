# BookSim policy

BookSim is a generic interconnection-network timing engine, not an NVIDIA
NVSwitch implementation. Pin and record its upstream SHA. Prefer an external
checkout/submodule or narrowly scoped adapter over copying the source. Keep
NVSwitch topology/protocol and NVLS multicast/reduction/completion semantics in
an explicitly named layer, with tests that can distinguish transport timing
from semantic completion.
