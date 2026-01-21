/**
 * Metamorphic Engine Header
 */

#ifndef METAMORPHIC_ENGINE_H
#define METAMORPHIC_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

// Mathematical model types
#define MODEL_LINEAR_REGRESSION 0
#define MODEL_NON_EUCLIDEAN_GEOMETRY 1
#define MODEL_MANIFOLD_LEARNING 2
#define MODEL_TOPOLOGICAL_DATA_ANALYSIS 3
#define MODEL_QUANTUM_WAVEFUNCTION 4
#define MODEL_CAUSAL_GRAPH 5
#define MODEL_HYPERBOLIC_GEOMETRY 6
#define MODEL_RIEMANNIAN_MANIFOLD 7

// Initialize the Metamorphic Engine
void metamorphic_engine_init();

// Register a strategy for profiling
void metamorphic_engine_register_strategy(const char* strategy_id, int model_type);

// Record a trade result
void metamorphic_engine_record_trade(const char* strategy_id, double pnl, int64_t timestamp_ns, int is_win);

// Get evolution status for a strategy
// Returns 1 if evolution is triggered, 0 otherwise
int metamorphic_engine_get_evolution_status(const char* strategy_id, int* current_model, int* proposed_model,
                                            double* alpha_decay_rate, int* evolution_cycle);

// Apply evolutionary model change
void metamorphic_engine_apply_evolution(const char* strategy_id);

// Get System Evolution Level (SEL) - overall system intelligence metric (0.0 to 1.0)
double metamorphic_engine_get_sel();

// Cleanup
void metamorphic_engine_cleanup();

#ifdef __cplusplus
}
#endif

#endif // METAMORPHIC_ENGINE_H
