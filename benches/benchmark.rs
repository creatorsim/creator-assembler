#![allow(clippy::unwrap_used)]
#![allow(clippy::missing_panics_doc)]

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use creator_assembler::parser::AST;
use creator_assembler::prelude::*;

static ARCH_JSON: &str = include_str!("arch.json");
static CODE: &str = include_str!("sample.s");
static NAME: &str = "sample.s";

fn parse(arch: &Architecture) -> AST {
    parser::parse(black_box(arch.config.comment_prefix), black_box(CODE))
        .map_err(|e| eprintln!("{}", e.render(NAME, CODE, true)))
        .unwrap()
}

pub fn benchmark_crate(c: &mut Criterion) {
    let arch = Architecture::from_json(ARCH_JSON).unwrap();
    c.bench_function("parse-only", |b| {
        b.iter(|| black_box(parse(&arch)));
    });

    let ast = parse(&arch);
    c.bench_function("compile-only", |b| {
        b.iter_batched(
            || ast.clone(),
            |ast| {
                black_box(compiler::compile(
                    black_box(&arch),
                    ast,
                    &black_box(0u8.into()),
                    black_box(HashMap::new()),
                    black_box(false),
                ))
                .map_err(|e| eprintln!("{}", e.render(NAME, CODE, true)))
                .unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("full-process", |b| {
        b.iter(|| {
            let ast = black_box(parse(&arch));
            black_box(compiler::compile(
                black_box(&arch),
                ast,
                &black_box(0u8.into()),
                black_box(HashMap::new()),
                black_box(false),
            ))
            .map_err(|e| eprintln!("{}", e.render(NAME, CODE, true)))
            .unwrap();
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(5)).measurement_time(Duration::from_secs(15));
    targets = benchmark_crate
}
criterion_main!(benches);
