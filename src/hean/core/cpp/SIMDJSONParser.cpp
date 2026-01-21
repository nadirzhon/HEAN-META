/**
 * SIMD-Optimized JSON Parser Implementation
 * Uses simdjson library for ultra-fast Bybit V5 message parsing
 */

#include "SIMDJSONParser.h"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// Use real simdjson library if available
#ifdef USE_REAL_SIMDJSON
#include <simdjson.h>
#define USE_SIMD_JSON
#endif

// memmem fallback for systems that don't have it
#ifndef _GNU_SOURCE
static inline void* memmem(const void* haystack, size_t haystacklen,
                           const void* needle, size_t needlelen) {
    if (needlelen == 0) return const_cast<void*>(haystack);
    if (needlelen > haystacklen) return nullptr;
    
    const char* h = static_cast<const char*>(haystack);
    const char* n = static_cast<const char*>(needle);
    
    for (size_t i = 0; i <= haystacklen - needlelen; ++i) {
        if (memcmp(h + i, n, needlelen) == 0) {
            return const_cast<char*>(h + i);
        }
    }
    return nullptr;
}
#endif

#ifdef USE_REAL_SIMDJSON
// Use simdjson library for ultra-fast parsing
static thread_local simdjson::dom::parser parser;
static thread_local simdjson::dom::element doc;

bool extract_json_double_simd(const char* json, size_t json_len, const char* field_name, double& output) {
    try {
        auto padded = simdjson::padded_string::load(json, json_len);
        auto element = parser.parse(padded);
        
        if (element.error()) {
            return false;
        }
        
        auto field_element = element[field_name];
        if (field_element.error()) {
            return false;
        }
        
        auto result = field_element.get<double>();
        if (result.error()) {
            return false;
        }
        
        output = result.value();
        return true;
    } catch (...) {
        return false;
    }
}

bool extract_json_string_simd(const char* json, size_t json_len, const char* field_name, char* output, size_t output_len) {
    try {
        auto padded = simdjson::padded_string::load(json, json_len);
        auto element = parser.parse(padded);
        
        if (element.error()) {
            return false;
        }
        
        auto field_element = element[field_name];
        if (field_element.error()) {
            return false;
        }
        
        auto result = field_element.get<std::string_view>();
        if (result.error()) {
            return false;
        }
        
        std::string_view sv = result.value();
        size_t copy_len = std::min(sv.size(), output_len - 1);
        std::memcpy(output, sv.data(), copy_len);
        output[copy_len] = '\0';
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_bybit_ticker_simd(const char* json, size_t json_len, BybitTickerData& output) {
    memset(&output, 0, sizeof(output));
    
    try {
        auto padded = simdjson::padded_string::load(json, json_len);
        auto element = parser.parse(padded);
        
        if (element.error()) {
            return false;
        }
        
        // Parse Bybit V5 ticker format: {"topic":"tickers.BTCUSDT","data":[{...}]}
        auto topic = element["topic"];
        auto data = element["data"];
        
        if (data.error()) {
            // Try direct parsing (single object)
            element.get<double>("lastPrice").tie(output.last_price);
            element.get<double>("bid1Price").tie(output.bid1_price);
            element.get<double>("ask1Price").tie(output.ask1_price);
            element.get<double>("volume24h").tie(output.volume_24h);
            
            auto ts = element["ts"];
            if (!ts.error()) {
                auto ts_val = ts.get<int64_t>();
                if (!ts_val.error()) {
                    output.timestamp = ts_val.value();
                }
            }
            
            auto sym = element["symbol"];
            if (!sym.error()) {
                auto sym_val = sym.get<std::string_view>();
                if (!sym_val.error()) {
                    std::string_view sv = sym_val.value();
                    size_t copy_len = std::min(sv.size(), sizeof(output.symbol) - 1);
                    std::memcpy(output.symbol, sv.data(), copy_len);
                    output.symbol[copy_len] = '\0';
                }
            }
        } else {
            // Array format - get first element
            auto first = data.at(0);
            if (first.error()) {
                return false;
            }
            
            first.get<double>("lastPrice").tie(output.last_price);
            first.get<double>("bid1Price").tie(output.bid1_price);
            first.get<double>("ask1Price").tie(output.ask1_price);
            first.get<double>("volume24h").tie(output.volume_24h);
            
            auto ts = first["ts"];
            if (!ts.error()) {
                auto ts_val = ts.get<int64_t>();
                if (!ts_val.error()) {
                    output.timestamp = ts_val.value();
                }
            }
            
            auto sym = first["symbol"];
            if (!sym.error()) {
                auto sym_val = sym.get<std::string_view>();
                if (!sym_val.error()) {
                    std::string_view sv = sym_val.value();
                    size_t copy_len = std::min(sv.size(), sizeof(output.symbol) - 1);
                    std::memcpy(output.symbol, sv.data(), copy_len);
                    output.symbol[copy_len] = '\0';
                }
            }
        }
        
        return output.last_price > 0.0;
    } catch (...) {
        return false;
    }
}

bool parse_bybit_orderbook_simd(const char* json, size_t json_len, BybitOrderbookData& output) {
    memset(&output, 0, sizeof(output));
    
    try {
        auto padded = simdjson::padded_string::load(json, json_len);
        auto element = parser.parse(padded);
        
        if (element.error()) {
            return false;
        }
        
        // Parse Bybit V5 orderbook format
        auto sym = element["s"];
        if (!sym.error()) {
            auto sym_val = sym.get<std::string_view>();
            if (!sym_val.error()) {
                std::string_view sv = sym_val.value();
                size_t copy_len = std::min(sv.size(), sizeof(output.symbol) - 1);
                std::memcpy(output.symbol, sv.data(), copy_len);
                output.symbol[copy_len] = '\0';
            }
        }
        
        auto u = element["u"];
        if (!u.error()) {
            auto u_val = u.get<int64_t>();
            if (!u_val.error()) {
                output.update_id = u_val.value();
            }
        }
        
        auto ts = element["ts"];
        if (!ts.error()) {
            auto ts_val = ts.get<int64_t>();
            if (!ts_val.error()) {
                output.timestamp = ts_val.value();
            }
        }
        
        // Parse bids array
        auto bids = element["b"];
        if (!bids.error()) {
            auto bids_array = bids.get_array();
            if (!bids_array.error()) {
                int idx = 0;
                for (auto bid : bids_array.value()) {
                    if (idx >= 25) break;
                    auto bid_array = bid.get_array();
                    if (!bid_array.error()) {
                        auto price = bid_array.at(0).get<double>();
                        auto qty = bid_array.at(1).get<double>();
                        if (!price.error() && !qty.error()) {
                            output.bids[idx][0] = price.value();
                            output.bids[idx][1] = qty.value();
                            idx++;
                        }
                    }
                }
                output.bid_count = idx;
            }
        }
        
        // Parse asks array
        auto asks = element["a"];
        if (!asks.error()) {
            auto asks_array = asks.get_array();
            if (!asks_array.error()) {
                int idx = 0;
                for (auto ask : asks_array.value()) {
                    if (idx >= 25) break;
                    auto ask_array = ask.get_array();
                    if (!ask_array.error()) {
                        auto price = ask_array.at(0).get<double>();
                        auto qty = ask_array.at(1).get<double>();
                        if (!price.error() && !qty.error()) {
                            output.asks[idx][0] = price.value();
                            output.asks[idx][1] = qty.value();
                            idx++;
                        }
                    }
                }
                output.ask_count = idx;
            }
        }
        
        return output.bid_count > 0 || output.ask_count > 0;
    } catch (...) {
        return false;
    }
}

#elif defined(USE_SIMD_JSON)

// SIMD-optimized string search for field names (fallback implementation)
static inline const char* simd_find_field(const char* json, size_t json_len, const char* field_name, size_t field_len) {
    // Simple SIMD-accelerated search (can be enhanced with proper SIMD intrinsics)
    const char* end = json + json_len - field_len;
    
    // Align to SIMD boundary (16 bytes for SSE, 32 bytes for AVX)
#ifdef __AVX2__
    const char* aligned_start = reinterpret_cast<const char*>(
        (reinterpret_cast<uintptr_t>(json) + 31) & ~31
    );
    
    // Use AVX2 for bulk comparison
    __m256i field_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(field_name));
    
    for (const char* p = json; p < end; p += 32) {
        __m256i data_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        __m256i cmp = _mm256_cmpeq_epi8(field_vec, data_vec);
        int mask = _mm256_movemask_epi8(cmp);
        
        if (mask != 0) {
            // Found potential match, verify with full string comparison
            if (memcmp(p, field_name, field_len) == 0) {
                // Check for JSON field pattern: "field": or ,"field":
                if ((p == json || p[-1] == ',' || p[-1] == '{') && p[field_len] == '"' && p[field_len+1] == ':') {
                    return p + field_len + 2;
                }
            }
        }
    }
#elif defined(__SSE4_2__)
    // SSE4.2 optimized search
    __m128i field_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(field_name));
    
    for (const char* p = json; p < end; p += 16) {
        __m128i data_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        __m128i cmp = _mm_cmpeq_epi8(field_vec, data_vec);
        int mask = _mm_movemask_epi8(cmp);
        
        if (mask == 0xFFFF && memcmp(p, field_name, field_len) == 0) {
            if ((p == json || p[-1] == ',' || p[-1] == '{') && p[field_len] == '"' && p[field_len+1] == ':') {
                return p + field_len + 2;
            }
        }
    }
#endif
    
    // Fallback to standard memmem
    const char* pos = static_cast<const char*>(memmem(json, json_len, field_name, field_len));
    while (pos) {
        if ((pos == json || pos[-1] == ',' || pos[-1] == '{') && pos[field_len] == '"' && pos[field_len+1] == ':') {
            return pos + field_len + 2;
        }
        pos = static_cast<const char*>(memmem(pos + field_len, end - pos - field_len, field_name, field_len));
    }
    
    return nullptr;
}

// Fast double parsing (optimized for JSON numbers)
static inline double fast_strtod(const char* str, char** endptr) {
    // Skip whitespace
    while (*str == ' ' || *str == '\t' || *str == '\n') {
        str++;
    }
    
    // Handle quoted strings
    if (*str == '"') {
        str++;
        double result = strtod(str, endptr);
        if (**endptr == '"') {
            (*endptr)++;
        }
        return result;
    }
    
    return strtod(str, endptr);
}

bool extract_json_double_simd(const char* json, size_t json_len, const char* field_name, double& output) {
    size_t field_len = strlen(field_name);
    const char* field_start = simd_find_field(json, json_len, field_name, field_len);
    
    if (!field_start) {
        return false;
    }
    
    // Skip whitespace and quotes
    const char* num_start = field_start;
    while (*num_start == ' ' || *num_start == '"' || *num_start == '\t') {
        num_start++;
    }
    
    char* endptr;
    output = fast_strtod(num_start, &endptr);
    
    return endptr != num_start && !isnan(output) && !isinf(output);
}

bool extract_json_string_simd(const char* json, size_t json_len, const char* field_name, char* output, size_t output_len) {
    size_t field_len = strlen(field_name);
    const char* field_start = simd_find_field(json, json_len, field_name, field_len);
    
    if (!field_start) {
        return false;
    }
    
    // Skip whitespace and opening quote
    const char* str_start = field_start;
    while (*str_start == ' ' || *str_start == '"' || *str_start == '\t') {
        if (*str_start == '"') {
            str_start++;
            break;
        }
        str_start++;
    }
    
    // Extract string until closing quote
    size_t i = 0;
    while (i < output_len - 1 && str_start[i] != '"' && str_start[i] != '\0' && str_start[i] != ',') {
        output[i] = str_start[i];
        i++;
    }
    output[i] = '\0';
    
    return i > 0;
}

bool parse_bybit_ticker_simd(const char* json, size_t json_len, BybitTickerData& output) {
    memset(&output, 0, sizeof(output));
    
    // Extract fields using SIMD-accelerated search
    extract_json_double_simd(json, json_len, "lastPrice", output.last_price);
    extract_json_double_simd(json, json_len, "bid1Price", output.bid1_price);
    extract_json_double_simd(json, json_len, "ask1Price", output.ask1_price);
    extract_json_double_simd(json, json_len, "volume24h", output.volume_24h);
    
    double ts_ms = 0.0;
    if (extract_json_double_simd(json, json_len, "ts", ts_ms)) {
        output.timestamp = static_cast<int64_t>(ts_ms);
    }
    
    extract_json_string_simd(json, json_len, "symbol", output.symbol, sizeof(output.symbol));
    
    return output.last_price > 0.0; // Valid if we have at least price
}

bool parse_bybit_orderbook_simd(const char* json, size_t json_len, BybitOrderbookData& output) {
    memset(&output, 0, sizeof(output));
    
    extract_json_string_simd(json, json_len, "s", output.symbol, sizeof(output.symbol));
    
    double u = 0.0;
    if (extract_json_double_simd(json, json_len, "u", u)) {
        output.update_id = static_cast<int64_t>(u);
    }
    
    double ts = 0.0;
    if (extract_json_double_simd(json, json_len, "ts", ts)) {
        output.timestamp = static_cast<int64_t>(ts);
    }
    
    // Parse bids array (simplified - in production, use proper array parsing)
    // For now, extract first bid
    double bid_price = 0.0, bid_qty = 0.0;
    if (extract_json_double_simd(json, json_len, "b", bid_price)) {
        // This is simplified - proper implementation needs array parsing
        output.bids[0][0] = bid_price;
        output.bid_count = 1;
    }
    
    return output.bid_count > 0 || output.ask_count > 0;
}

#else
// Fallback implementation without SIMD
bool extract_json_double_simd(const char* json, size_t json_len, const char* field_name, double& output) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":", field_name);
    const char* pos = strstr(json, pattern);
    if (!pos) return false;
    
    const char* value_start = pos + strlen(pattern);
    while (*value_start == ' ' || *value_start == '"') value_start++;
    
    char* endptr;
    output = strtod(value_start, &endptr);
    return endptr != value_start;
}

bool extract_json_string_simd(const char* json, size_t json_len, const char* field_name, char* output, size_t output_len) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":\"", field_name);
    const char* pos = strstr(json, pattern);
    if (!pos) return false;
    
    const char* str_start = pos + strlen(pattern);
    size_t i = 0;
    while (i < output_len - 1 && str_start[i] != '"' && str_start[i] != '\0') {
        output[i] = str_start[i];
        i++;
    }
    output[i] = '\0';
    return i > 0;
}

bool parse_bybit_ticker_simd(const char* json, size_t json_len, BybitTickerData& output) {
    memset(&output, 0, sizeof(output));
    extract_json_double_simd(json, json_len, "lastPrice", output.last_price);
    extract_json_double_simd(json, json_len, "bid1Price", output.bid1_price);
    extract_json_double_simd(json, json_len, "ask1Price", output.ask1_price);
    return output.last_price > 0.0;
}

bool parse_bybit_orderbook_simd(const char* json, size_t json_len, BybitOrderbookData& output) {
    memset(&output, 0, sizeof(output));
    extract_json_string_simd(json, json_len, "s", output.symbol, sizeof(output.symbol));
    return true;
}
#endif
