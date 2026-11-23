# HCA image: one binary on a minimal base, no shell or package manager.
# Build from the repository root:
#   docker build -t hca -f docker/hca.Dockerfile .

FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml clippy.toml hca.example.toml ./
COPY crates ./crates
RUN cargo build --release --locked -p hca

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /src/target/release/hca /usr/local/bin/hca
COPY hca.example.toml /etc/hca/hca.toml
# Runs as uid 1000 so workspace files match the sandbox's default user.
USER 1000:1000
ENV HCA_CONFIG=/etc/hca/hca.toml HCA_BIND=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/hca"]
CMD ["serve"]
