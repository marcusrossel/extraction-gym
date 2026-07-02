mod extract;

pub use extract::*;

use egraph_serialize::*;

use indexmap::IndexMap;
use ordered_float::NotNan;

use anyhow::Context;

use std::io::Write;
use std::path::PathBuf;

pub type Cost = NotNan<f64>;
pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(f64::INFINITY) };

#[derive(PartialEq, Eq)]
enum Optimal {
    Tree,
    #[cfg(feature = "ilp-cbc")]
    Dag,
    Neither,
}

struct ExtractorDetail {
    extractor: Box<dyn Extractor>,
    #[cfg_attr(not(test), allow(dead_code))]
    optimal: Optimal,
    use_for_bench: bool,
}

fn extractors() -> IndexMap<&'static str, ExtractorDetail> {
    [
        (
            "bottom-up",
            ExtractorDetail {
                extractor: extract::bottom_up::BottomUpExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "faster-bottom-up",
            ExtractorDetail {
                extractor: extract::faster_bottom_up::FasterBottomUpExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "prio-queue",
            ExtractorDetail {
                extractor: extract::prio_queue::PrioQueueExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "reach-prio-queue",
            ExtractorDetail {
                extractor: extract::reach_prio_queue::ReachPrioQueueExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "naive-a-star",
            ExtractorDetail {
                extractor: extract::naive_a_star::NaiveAStarExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "a-star",
            ExtractorDetail {
                extractor: extract::a_star::AStarExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "faster-greedy-dag",
            ExtractorDetail {
                extractor: extract::faster_greedy_dag::FasterGreedyDagExtractor.boxed(),
                optimal: Optimal::Neither,
                use_for_bench: true,
            },
        ),
        /*(
            "global-greedy-dag",
            ExtractorDetail {
                extractor: extract::global_greedy_dag::GlobalGreedyDagExtractor.boxed(),
                optimal: Optimal::Neither,
                use_for_bench: true,
            },
        ),*/
        #[cfg(feature = "ilp-cbc")]
        (
            "ilp-cbc-timeout",
            ExtractorDetail {
                extractor: extract::ilp_cbc::CbcExtractorWithTimeout::<10>.boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-cbc")]
        (
            "ilp-cbc",
            ExtractorDetail {
                extractor: extract::ilp_cbc::CbcExtractor.boxed(),
                optimal: Optimal::Dag,
                use_for_bench: false, // takes >10 hours sometimes
            },
        ),
        #[cfg(feature = "ilp-cbc")]
        (
            "faster-ilp-cbc-timeout",
            ExtractorDetail {
                extractor: extract::faster_ilp_cbc::FasterCbcExtractorWithTimeout::<10>.boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-cbc")]
        (
            "faster-ilp-cbc",
            ExtractorDetail {
                extractor: extract::faster_ilp_cbc::FasterCbcExtractor.boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
    ]
    .into_iter()
    .collect()
}

// A `Benchmark` collects all of the information necessary to run a single given extractor on a
// single given e-graph (see `Benchmark::run`).
struct Benchmark<'a> {
    extractor_name:   String,
    extractor_detail: &'a ExtractorDetail,
    egraph:           EGraph,
    out_filename:     PathBuf,
    filename:         String,
}

impl<'a> Benchmark<'a> {
    fn execute(&self) -> String {
        let egraph = &self.egraph;
        let roots = &egraph.root_eclasses;

        let start_time = std::time::Instant::now();
        let result = self.extractor_detail.extractor.extract(egraph, roots);
        let us = start_time.elapsed().as_micros();

        result.check(egraph);

        let tree = result.tree_cost(egraph, roots);
        let dag = result.dag_cost(egraph, roots);

        let filename = &self.filename;
        let extractor_name = &self.extractor_name;
        let roots_str = roots.iter().map(|r| format!("\"{r}\"")).collect::<Vec<_>>().join(", ");
        log::info!("{filename:40}\t{extractor_name:10}\t{tree:5}\t{dag:5}\t{us:5}");

        format!(r#"    {{
      "roots": [{roots_str}],
      "tree": {tree},
      "dag": {dag},
      "micros": {us}
    }}"#)
    }

    fn write_results(&self, entries: &[String]) {
        let mut out_file = std::fs::File::create(&self.out_filename).unwrap();
        let entries_str = entries.join(",\n");
        let name = &self.filename;
        let extractor = &self.extractor_name;
        writeln!(
            out_file,
            r#"{{
  "name": "{name}",
  "extractor": "{extractor}",
  "results": [
{entries_str}
  ]
}}"#
        ).unwrap();
    }

    fn run(&self) {
        let entry = self.execute();
        self.write_results(&[entry]);
    }

    // Runs a `Benchmark` by splitting it into multiple single-root benchmarks and writing all
    // results to a single output file.
    fn run_as_single_root(mut self) {
        if self.egraph.root_eclasses.len() <= 1 {
            self.run();
        } else {
            let roots = std::mem::take(&mut self.egraph.root_eclasses);
            let mut entries = Vec::new();

            for root in roots.iter() {
                self.egraph.root_eclasses = vec![root.clone()];
                entries.push(self.execute());
            }

            self.write_results(&entries);
        }
    }
}

fn main() {
    env_logger::init();

    let mut extractors = extractors();
    extractors.retain(|_, ed| ed.use_for_bench);

    let mut args = pico_args::Arguments::from_env();

    let extractor_name: String = args
        .opt_value_from_str("--extractor")
        .unwrap()
        .unwrap_or_else(|| "bottom-up".into());

    if extractor_name == "print" {
        for name in extractors.keys() {
            println!("{}", name);
        }
        return;
    }

    let single_root: bool = args.contains("--single-root");

    let out_filename: PathBuf = args
        .opt_value_from_str("--out")
        .unwrap()
        .unwrap_or_else(|| "out.json".into());

    let filename: String = args.free_from_str().unwrap();

    let rest = args.finish();
    if !rest.is_empty() {
        panic!("Unknown arguments: {:?}", rest);
    }

    let egraph = EGraph::from_json_file(&filename)
        .with_context(|| format!("Failed to parse {filename}"))
        .unwrap();

    let extractor_detail = extractors
        .get(extractor_name.as_str())
        .with_context(|| format!("Unknown extractor: {extractor_name}"))
        .unwrap();

    let benchmark = Benchmark { extractor_name, extractor_detail, egraph, out_filename, filename };

    // If the `--single-root` flag is passed, run each benchmark with only a single root e-class.
    // For benchmarks which would have had multiple root e-classes, we break it into multiple
    // single-root benchmarks.
    if single_root {
        benchmark.run_as_single_root();
    } else {
        benchmark.run();
    }
}

#[cfg(test)]
pub mod test;
