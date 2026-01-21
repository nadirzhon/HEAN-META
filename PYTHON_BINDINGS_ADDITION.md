# Python Bindings Addition Guide

## Required Changes to python_bindings.cpp

Since `python_bindings.cpp` is large, add these bindings at the end of the `PYBIND11_MODULE` section (before the final closing brace).

### 1. Add Includes (at top of file, after existing includes)

```cpp
#include "swarm_manager.h"
#include "ofi_monitor.h"
```

### 2. Add Bindings (before the final `}` of PYBIND11_MODULE)

```cpp
    // Swarm Manager bindings
    py::class_<SwarmManager>(m, "SwarmManager")
        .def(py::init<int, double>(), 
             py::arg("num_agents") = 100, 
             py::arg("consensus_threshold") = 0.80)
        .def("initialize_swarm", &SwarmManager::initialize_swarm, 
             "Initialize swarm with specialized agents")
        .def("update_orderflow", &SwarmManager::update_orderflow, 
             py::arg("symbol"), py::arg("snapshot"), 
             "Update orderflow features for all agents")
        .def("get_consensus", &SwarmManager::get_consensus,
             py::arg("symbol"), "Get consensus decision (Fast-Voting)")
        .def("get_consensus_confidence", &SwarmManager::get_consensus_confidence,
             py::arg("symbol"), "Get consensus confidence level (0.0 to 1.0)")
        .def("get_agent_distribution", &SwarmManager::get_agent_distribution,
             "Get agent distribution by type")
        .def("reset_agents", &SwarmManager::reset_agents, 
             "Reset agents for new trading session");
    
    // OrderflowSnapshot binding
    py::class_<OrderflowSnapshot>(m, "OrderflowSnapshot")
        .def(py::init<>())
        .def_readwrite("delta", &OrderflowSnapshot::delta)
        .def_readwrite("ofi", &OrderflowSnapshot::ofi)
        .def_readwrite("vpin", &OrderflowSnapshot::vpin)
        .def_readwrite("micro_momentum", &OrderflowSnapshot::micro_momentum)
        .def_readwrite("price", &OrderflowSnapshot::price)
        .def_readwrite("bid_volume", &OrderflowSnapshot::bid_volume)
        .def_readwrite("ask_volume", &OrderflowSnapshot::ask_volume)
        .def_readwrite("timestamp_ns", &OrderflowSnapshot::timestamp_ns);
    
    // ConsensusResult binding
    py::class_<ConsensusResult>(m, "ConsensusResult")
        .def(py::init<>())
        .def_readwrite("consensus", &ConsensusResult::consensus)
        .def_readwrite("buy_vote_percentage", &ConsensusResult::buy_vote_percentage)
        .def_readwrite("sell_vote_percentage", &ConsensusResult::sell_vote_percentage)
        .def_readwrite("average_confidence", &ConsensusResult::average_confidence)
        .def_readwrite("total_agents", &ConsensusResult::total_agents)
        .def_readwrite("buy_votes", &ConsensusResult::buy_votes)
        .def_readwrite("sell_votes", &ConsensusResult::sell_votes)
        .def_readwrite("consensus_reached", &ConsensusResult::consensus_reached)
        .def_readwrite("execution_signal_strength", &ConsensusResult::execution_signal_strength);
    
    py::enum_<AgentDecision>(m, "AgentDecision")
        .value("BUY", AgentDecision::BUY)
        .value("SELL", AgentDecision::SELL)
        .value("NEUTRAL", AgentDecision::NEUTRAL)
        .export_values();
    
    // OFI Monitor bindings
    py::class_<OFIMonitor>(m, "OFIMonitor")
        .def(py::init<int, double, bool>(), 
             py::arg("lookback_window") = 20, 
             py::arg("price_level_size") = 0.01,
             py::arg("use_ml") = true)
        .def("update_orderbook", [](OFIMonitor& self, 
             const std::string& symbol,
             const std::vector<std::pair<double, double>>& bids,
             const std::vector<std::pair<double, double>>& asks,
             int64_t timestamp_ns) {
            self.update_orderbook(symbol, bids, asks, timestamp_ns);
        }, py::arg("symbol"), py::arg("bids"), py::arg("asks"), py::arg("timestamp_ns"),
           "Update orderbook data")
        .def("update_trade", &OFIMonitor::update_trade,
             py::arg("symbol"), py::arg("price"), py::arg("size"), 
             py::arg("is_buy"), py::arg("timestamp_ns"),
             "Update trade data for delta calculation")
        .def("get_ofi", &OFIMonitor::get_ofi,
             py::arg("symbol"), "Get current OFI for symbol")
        .def("predict_next_ticks", &OFIMonitor::predict_next_ticks,
             py::arg("symbol"), py::arg("current_price"), "Predict next 3 ticks")
        .def("get_price_level_ofi", &OFIMonitor::get_price_level_ofi,
             py::arg("symbol"), py::arg("price"), "Get OFI at specific price level")
        .def("get_delta", &OFIMonitor::get_delta,
             py::arg("symbol"), "Get delta (net buy - sell volume)")
        .def("reset_symbol", &OFIMonitor::reset_symbol,
             py::arg("symbol"), "Reset data for symbol")
        .def("set_ml_prediction", &OFIMonitor::set_ml_prediction,
             py::arg("enabled"), "Enable/disable ML prediction")
        .def("load_model", &OFIMonitor::load_model,
             py::arg("model_path"), "Load ML model");
    
    // OFIResult binding
    py::class_<OFIResult>(m, "OFIResult")
        .def(py::init<>())
        .def_readwrite("ofi_value", &OFIResult::ofi_value)
        .def_readwrite("delta", &OFIResult::delta)
        .def_readwrite("buy_pressure", &OFIResult::buy_pressure)
        .def_readwrite("sell_pressure", &OFIResult::sell_pressure)
        .def_readwrite("imbalance_strength", &OFIResult::imbalance_strength)
        .def_readwrite("price_level_ofi", &OFIResult::price_level_ofi);
    
    // PricePrediction binding
    py::class_<PricePrediction>(m, "PricePrediction")
        .def(py::init<>())
        .def_readwrite("predicted_prices", &PricePrediction::predicted_prices)
        .def_readwrite("probabilities", &PricePrediction::probabilities)
        .def_readwrite("overall_confidence", &PricePrediction::overall_confidence)
        .def_readwrite("is_bullish", &PricePrediction::is_bullish)
        .def_readwrite("expected_movement", &PricePrediction::expected_movement)
        .def_readwrite("accuracy_estimate", &PricePrediction::accuracy_estimate);
```

### 3. Build Instructions

After adding the bindings, rebuild the C++ extension:

```bash
cd src/hean/core/cpp
mkdir -p build
cd build
cmake ..
make
```

The compiled module will be at `build/graph_engine_py*.so` (or `.pyd` on Windows).

### 4. Testing the Bindings

```python
import sys
sys.path.insert(0, 'src/hean/core/cpp/build')
import graph_engine_py

# Test SwarmManager
swarm = graph_engine_py.SwarmManager(num_agents=100, consensus_threshold=0.80)
swarm.initialize_swarm()

# Test OFIMonitor
ofi = graph_engine_py.OFIMonitor(lookback_window=20, price_level_size=0.01, use_ml=True)
```

## Note on Vector Pairs for orderbook

The `update_orderbook` method uses `std::vector<std::pair<double, double>>` for bids/asks. 
If pybind11 has issues with this, you may need to use a wrapper or convert to separate vectors:

```cpp
.def("update_orderbook", [](OFIMonitor& self, 
     const std::string& symbol,
     const py::list& bids,  // Python list of [price, size] pairs
     const py::list& asks,
     int64_t timestamp_ns) {
    std::vector<std::pair<double, double>> bid_vec, ask_vec;
    
    for (auto item : bids) {
        py::list pair = py::cast<py::list>(item);
        bid_vec.push_back({py::cast<double>(pair[0]), py::cast<double>(pair[1])});
    }
    
    for (auto item : asks) {
        py::list pair = py::cast<py::list>(item);
        ask_vec.push_back({py::cast<double>(pair[0]), py::cast<double>(pair[1])});
    }
    
    self.update_orderbook(symbol, bid_vec, ask_vec, timestamp_ns);
}, ...)
```
