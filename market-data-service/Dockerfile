# Multi-stage build for minimal image size

# Build stage
FROM rust:1.75-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache musl-dev openssl-dev pkgconfig

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Cache dependencies by building dummy project
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy source code
COPY src ./src

# Build application
RUN cargo build --release

# Final stage
FROM alpine:latest

RUN apk add --no-cache libgcc ca-certificates

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/market-data-service .

# Create non-root user
RUN adduser -D -u 1000 heanmeta && \
    chown -R heanmeta:heanmeta /app

USER heanmeta

# Expose ports
EXPOSE 8080 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:8080/health || exit 1

CMD ["./market-data-service"]
