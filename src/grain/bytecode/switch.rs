use crate::{ast::RangeCase, Dynamic, INT};
#[cfg(feature = "no_std")]
use std::prelude::v1::*;

/// A `switch`, as one dispatch table.
///
/// Matching a case is *hash* equality, not `==`. That distinction is Rhai's
/// and it is visible: `switch 1 { 1.0 => .. }` does not match, because an
/// integer and a float hash differently, while `1 == 1.0` is true.
///
/// ## Why the hashes travel, and what that costs
///
/// Rhai's parser keeps only the hash of each case — the value itself is not in
/// the AST (`ast/stmt.rs:336`), so there is nothing to re-hash later. The
/// hashes have to be written out as they are.
///
/// And by default they do not survive the trip: `get_hasher` falls back to
/// `ahash::AHasher::default()`, and Rhai's default features include
/// `ahash/runtime-rng`, so the seed is drawn per process. Rhai gets away with
/// baking hashes into its AST only because it parses and evaluates in one.
///
/// So an artifact containing a `switch` requires
/// [`rhai::config::hashing::set_hashing_seed`] to have been called with the
/// same seed on both sides. That is not something the format can enforce, but
/// it is something it can *check*: [`probe`] hashes a fixed value, the
/// artifact carries the result, and a loader that computes a different one
/// refuses rather than dispatching every case to the default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Switch {
    /// One entry per distinct case value, ascending by hash. The target is
    /// the head of that value's chain of guarded arms.
    ///
    /// Ordered by hash rather than by source order because
    /// [`Switch::case_target`] bisects it when [`switch_index`] says to. The
    /// compiler sorts the groups it emits and the reader sorts what it reads,
    /// so nothing that reaches a lookup is unsorted.
    ///
    /// Private because `index` holds a copy of every target in it, and a
    /// rewrite that reached one and not the other would send subjects to an
    /// address that stopped being their arm. [`Switch::retarget`] is how a
    /// target changes; [`Switch::cases`] is how the writer and the verifier
    /// read one.
    cases: Vec<SwitchCase>,
    /// Checked only when no case matched in this table.
    ///
    /// Disjoint and in ascending order, which Rhai's are not: the compiler
    /// splits overlapping arms apart so that the first entry containing a
    /// value is the only one that can match it. See `compile::cases`.
    pub ranges: Vec<SwitchRange>,
    /// Where to go when nothing matched. Always present: an absent `_` arm
    /// compiles to a jump past the statement.
    pub default: u32,
    /// Where a hash sends control: an open-addressed table with linear
    /// probing, empty only for a table with no cases to index.
    ///
    /// Derived, not stored. The hashes it keys on are already in the artifact,
    /// so this is rebuilt at load and the wire format does not know it exists.
    /// Private for the same reason [`Self::new`] is the only constructor: an
    /// index nobody can supply is an index that cannot arrive disagreeing with
    /// the cases it describes.
    index: Box<[Bucket]>,
}

/// One slot of a [`Switch`]'s case index.
///
/// Sixteen bytes, so reading one is a single aligned load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Bucket {
    /// The hash of the case this slot holds, and zero in an empty slot.
    ///
    /// Kept here as well as in `cases` so that a subject no case names is
    /// turned away by the one load — which is every subject that goes on to
    /// the range arms or the default.
    hash: u64,
    /// Where that case sends control, and `None` in an empty slot.
    ///
    /// The target itself rather than a position in `cases`, so a lookup that
    /// hits reads this slot and nothing else. Reaching `cases` for the target
    /// would be a second dependent load into a second allocation, through the
    /// `Vec` header to find it. What it costs is that a target is written down
    /// twice, which is why only [`Switch::retarget`] may move one.
    ///
    /// `None` rather than a reserved target value: every `u32` is a legal
    /// address in a chunk, so occupancy has nowhere to hide inside `target`.
    /// It is free anyway — the four bytes the discriminant takes are the four
    /// this struct was padding with.
    target: Option<u32>,
}

impl Bucket {
    /// The index over `cases`, empty for a `cases` with nothing in it.
    ///
    /// Every table with a case gets one. A slot carries its own target, so a
    /// probe is a single load whatever the table's size, and there is no size
    /// at which walking `cases` instead is fewer: one entry costs the same
    /// load either way, and every entry after the first costs the scan another
    /// comparison and the probe nothing.
    ///
    /// Built once, by the compiler or by the reader, so nothing here is on a
    /// hot path.
    fn index(cases: &[SwitchCase]) -> Box<[Bucket]> {
        if cases.is_empty() {
            return Box::default();
        }

        // Twice the entries and rounded up, so the slot a hash lands in is a
        // mask rather than a division. Half full is also what keeps the walks
        // short — linear probing lengthens sharply above that — and it is what
        // guarantees the empty slot that ends a walk exists at all.
        let capacity = match cases
            .len()
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
        {
            Some(capacity) => capacity,
            // `usize` is not always wider than the count it holds: on a 32-bit
            // target a `cases` past half the address space cannot be doubled.
            // Answered with no index, which `Switch::case_target` reads as a
            // table to scan.
            None => return Box::default(),
        };

        let mut index = vec![
            Self {
                hash: 0,
                target: None,
            };
            capacity
        ];
        let mask = capacity - 1;
        for case in cases {
            let mut at = (case.hash as usize) & mask;
            loop {
                let slot = &mut index[at];
                if slot.target.is_none() {
                    *slot = Self {
                        hash: case.hash,
                        target: Some(case.target),
                    };
                    break;
                }
                // A hash already in the table keeps the slot it has, which
                // holds the earlier case's target. Inserting in order is what
                // makes a lookup answer with the first entry holding a hash,
                // the way bisecting to the partition point does.
                if slot.hash == case.hash {
                    break;
                }
                at = (at + 1) & mask;
            }
        }
        index.into_boxed_slice()
    }
}

/// One `value => ...` arm, keyed by Rhai's hash of the value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchCase {
    /// Rhai's hash of the case value.
    pub hash: u64,
    /// Where to jump to.
    pub target: u32,
}

/// One `a..b => ...` arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchRange {
    /// The lower bound
    pub from: INT,
    /// The upper bound
    pub to: INT,
    /// Whether the upper bound is included.
    pub inclusive: bool,
    /// Where to jump to.
    pub target: u32,
}

impl SwitchRange {
    /// Whether a subject falls in this range.
    ///
    /// Delegates to Rhai's own `RangeCase` rather than comparing integers,
    /// because a range arm matches more than integers: `switch 5.5 { 0..10 =>
    /// .. }` matches, and under the `decimal` feature so does a `Decimal`
    /// (`ast/stmt.rs:254`). Rebuilding the case is two moves and no
    /// allocation, and it means there is one definition of what a range arm
    /// covers.
    #[must_use]
    pub fn contains(&self, value: &Dynamic) -> bool {
        let case: RangeCase = if self.inclusive {
            (self.from..=self.to).into()
        } else {
            (self.from..self.to).into()
        };
        case.contains(value)
    }
}

impl Switch {
    /// A table, and the index over the cases in it.
    ///
    /// The only constructor, so there is no `Switch` whose index was built
    /// from some other `cases` than the one it is holding. Built whatever
    /// [`switch_index`] currently says, because that answer can change between
    /// here and the dispatch that reads it.
    #[must_use]
    pub fn new(cases: Vec<SwitchCase>, ranges: Vec<SwitchRange>, default: u32) -> Self {
        let index = Bucket::index(&cases);
        Self {
            cases,
            ranges,
            default,
            index,
        }
    }

    /// The hashed arms, ascending by hash.
    ///
    /// What the writer puts on the wire and the verifier walks for targets.
    /// Read-only: the index holds a copy of each target, so a rewrite goes
    /// through [`Self::retarget`].
    #[must_use]
    #[inline]
    pub fn cases(&self) -> &[SwitchCase] {
        &self.cases
    }

    /// Rewrite every target in this table, and rebuild the index over them.
    ///
    /// One pass over the case arms, the range arms and the default, in that
    /// order. The rebuild is what makes this the only way a target may change:
    /// a case's target is written down twice, and the copy a dispatch reads is
    /// the one in the index.
    ///
    /// # Errors
    ///
    /// Whatever `resolve` refuses, at the first target it refuses.
    pub fn retarget<E>(
        &mut self,
        mut resolve: impl FnMut(&mut u32) -> Result<(), E>,
    ) -> Result<(), E> {
        let walked = self
            .cases
            .iter_mut()
            .map(|case| &mut case.target)
            .chain(self.ranges.iter_mut().map(|range| &mut range.target))
            .chain(core::iter::once(&mut self.default))
            .try_for_each(&mut resolve);

        // Rebuilt whether or not the walk finished. Nothing runs a table whose
        // rewrite failed, but an index left describing half a rewrite is a
        // wrong arm waiting for somebody to keep the table anyway.
        self.index = Bucket::index(&self.cases);
        walked
    }

    /// Where a subject sends control.
    ///
    /// The order is Rhai's table order (`eval/stmt.rs:517-564`): reject
    /// non-hashable subjects, then try hashed cases, then ranges, then default.
    #[must_use]
    pub fn dispatch(&self, subject: &Dynamic) -> u32 {
        // Hashing an non-hashable value panics, so this is a guard and not an
        // optimization.
        if !subject.is_hashable() {
            return self.default;
        }

        if let Some(target) = self.case_target(hash_of(subject)) {
            return target;
        }

        // Disjoint, so the first containing entry is the only one.
        if let Some(range) = self.ranges.iter().find(|r| r.contains(subject)) {
            return range.target;
        }

        self.default
    }

    /// Where the first case naming `hash` sends control, if one does.
    ///
    /// Whichever spelling [`switch_index`] names. They answer alike, which is
    /// what makes the choice a measurement rather than a change: both take the
    /// *first* entry of `cases` holding the hash, so a table that somehow
    /// holds one twice picks the same arm either way. Both are `u64` equality
    /// on what Rhai hashed; neither is `==` on the value
    /// (`eval/stmt.rs:517-564`).
    #[inline]
    fn case_target(&self, hash: u64) -> Option<u32> {
        self.case_target_indexed(hash)
    }

    /// [`Self::case_target`] read out of the index.
    ///
    /// A fixed number of steps rather than one per doubling of the table: mask
    /// the hash, read the slot, and walk forward only past slots some other
    /// hash landed in. A hit reads that slot and nothing else — the target is
    /// in it — and a miss usually reads only it too, because the slot carries
    /// the whole hash and the table is at most half full, so the slot a hash
    /// lands in is empty at least half the time. Either way `cases` is not
    /// touched, so no `Vec` is turned into a slice to answer.
    ///
    /// A table with no index is a table with no cases, and scanning it is how
    /// that answers nothing.
    fn case_target_indexed(&self, hash: u64) -> Option<u32> {
        if self.index.is_empty() {
            return self
                .cases
                .iter()
                .find(|case| case.hash == hash)
                .map(|case| case.target);
        }

        let mask = self.index.len() - 1;
        // The low bits. `func::get_hasher` is ahash, whose `finish` carries
        // every input bit into every output one, so no end of it is the better
        // end to take. Nothing rests on that being so: the slot holds the
        // whole hash and that is what decides a match, so a hasher with worse
        // low bits would cost walk length rather than correctness.
        let mut at = (hash as usize) & mask;
        // A walk cannot pass an empty slot and one always exists, so this
        // cannot go round — but counting the slots makes that a property of
        // the loop rather than of an argument made where the table was built.
        for _ in 0..self.index.len() {
            // Masked to the length, so `get` can only be `Some`. It is spelled
            // as a check anyway because that is the cost of never having to
            // ask whether a corrupt table could reach here: one comparison
            // against a length already in a register.
            let slot = match self.index.get(at) {
                Some(slot) => slot,
                None => break,
            };
            // An empty slot ends the walk: nothing past it landed here.
            let target = match slot.target {
                Some(target) => target,
                None => break,
            };
            if slot.hash == hash {
                return Some(target);
            }
            at = (at + 1) & mask;
        }
        None
    }

    /// [`Self::case_target`] bisected, which is what `cases` being ascending
    /// by hash is for, and what [`Self::case_target_indexed`] is checked
    /// against: the index is a second copy of the case targets, and a table
    /// that answers with the wrong arm is a wrong answer rather than a
    /// failure.
    ///
    /// `partition_point` rather than `binary_search_by_key` so that a table
    /// holding one hash twice answers with the first of them, which is what
    /// the indexed spelling does.
    #[cfg(test)]
    fn case_target_bisected(&self, hash: u64) -> Option<u32> {
        let at = self.cases.partition_point(|case| case.hash < hash);
        self.cases
            .get(at)
            .filter(|case| case.hash == hash)
            .map(|case| case.target)
    }
}

/// A fixed value hashed with the engine's hasher, so two processes can find
/// out whether their case hashes mean the same thing.
///
/// Not a checksum of the seed — the seed is not readable as a number the
/// format could compare. This is the observable consequence of it.
#[must_use]
pub fn probe() -> u64 {
    hash_of(&Dynamic::from("rhaigrain switch probe"))
}

fn hash_of(value: &Dynamic) -> u64 {
    use core::hash::{Hash, Hasher};

    let mut hasher = crate::func::get_hasher();
    value.hash(&mut hasher);
    hasher.finish()
}

/// The hash Rhai's `switch` would key `value` under.
///
/// Test-only. The compiler never hashes anything: Rhai's parser has already
/// grouped the arms by hash, and the hashes are all it kept.
#[cfg(test)]
fn case_hash(value: &Dynamic) -> Option<u64> {
    value.is_hashable().then(|| hash_of(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A script integer, as a subject or a case.
    ///
    /// Spelled through `INT` rather than as an `i64` literal because `only_i32`
    /// narrows it: a `Dynamic` built from the wider type there is a boxed host
    /// value, which has no hash and so matches nothing.
    fn int(value: INT) -> Dynamic {
        Dynamic::from(value)
    }

    /// A host type, which has no hash.
    #[derive(Debug, Clone)]
    struct Opaque;

    /// Sorted, because both the compiler and the reader sort: a table built
    /// any other way is not one they could have produced.
    fn table(cases: &[(&Dynamic, u32)], ranges: Vec<SwitchRange>, default: u32) -> Switch {
        let mut cases: Vec<SwitchCase> = cases
            .iter()
            .filter_map(|(value, target)| {
                Some(SwitchCase {
                    hash: case_hash(value)?,
                    target: *target,
                })
            })
            .collect();
        cases.sort_by_key(|case| case.hash);
        Switch::new(cases, ranges, default)
    }

    /// A table of `count` distinct hashes, each arm named by its own target.
    ///
    /// The hashes are spread by the golden ratio rather than counted up, so
    /// that the slots a table of them fills are not consecutive and the walks
    /// past occupied slots are real.
    fn spread(count: u64) -> Vec<SwitchCase> {
        let mut cases: Vec<SwitchCase> = (0..count)
            .map(|i| SwitchCase {
                hash: i.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                target: 100 + i as u32,
            })
            .collect();
        cases.sort_by_key(|case| case.hash);
        cases
    }

    #[test]
    fn a_matching_case_wins() {
        let (one, two) = (int(1), int(2));
        let table = table(&[(&one, 10), (&two, 20)], Vec::new(), 99);

        assert_eq!(table.dispatch(&one), 10);
        assert_eq!(table.dispatch(&two), 20);
        assert_eq!(table.dispatch(&int(3)), 99);
    }

    /// The distinction that makes this hashing rather than `==`: Rhai does not
    /// match an integer against a float case, even though `1 == 1.0`.
    #[cfg(not(feature = "no_float"))]
    #[test]
    fn a_float_does_not_match_an_integer_case() {
        let one = int(1);
        let table = table(&[(&one, 10)], Vec::new(), 99);

        let float = Dynamic::from(1.0 as crate::FLOAT);
        // The subject has to reach the hasher for this to say anything. Built
        // from a literal `f64` it would not under `f32_float`; that is a boxed
        // host value, and it would land on the default for having no hash at
        // all rather than for hashing differently.
        assert!(float.is_hashable(), "this test needs a hashable float");
        assert_eq!(table.dispatch(&float), 99);
    }

    #[test]
    fn strings_and_characters_match_by_value() {
        let (text, ch, flag) = (
            Dynamic::from("hello"),
            Dynamic::from('x'),
            Dynamic::from(true),
        );
        let table = table(&[(&text, 10), (&ch, 20), (&flag, 30)], Vec::new(), 99);

        assert_eq!(table.dispatch(&Dynamic::from("hello")), 10);
        assert_eq!(table.dispatch(&Dynamic::from("other")), 99);
        assert_eq!(table.dispatch(&ch), 20);
        assert_eq!(table.dispatch(&flag), 30);
    }

    /// A range arm covers the reals between its bounds, not just the integers
    /// in them — which is why the check delegates to Rhai's own `RangeCase`
    /// rather than comparing integers.
    #[test]
    #[cfg(not(feature = "no_float"))]
    fn a_range_catches_a_float_between_its_bounds() {
        let table = table(
            &[],
            vec![SwitchRange {
                from: 0,
                to: 10,
                inclusive: false,
                target: 20,
            }],
            99,
        );

        // The script float type, not `f64`: under `f32_float` a `Dynamic`
        // holding an `f64` is a foreign type and never matches a range arm.
        assert_eq!(table.dispatch(&Dynamic::from(5.5 as rhai::FLOAT)), 20);
        assert_eq!(
            table.dispatch(&Dynamic::from(10.0 as rhai::FLOAT)),
            99,
            "exclusive end"
        );
        assert_eq!(table.dispatch(&Dynamic::from(-0.5 as rhai::FLOAT)), 99);
    }

    /// Ranges are consulted only after the cases miss.
    #[test]
    fn a_range_catches_what_no_case_did() {
        let one = int(1);
        let table = table(
            &[(&one, 10)],
            vec![
                SwitchRange {
                    from: 5,
                    to: 8,
                    inclusive: false,
                    target: 20,
                },
                SwitchRange {
                    from: 8,
                    to: 10,
                    inclusive: true,
                    target: 30,
                },
            ],
            99,
        );

        assert_eq!(table.dispatch(&one), 10, "a case still wins");
        assert_eq!(table.dispatch(&int(5)), 20);
        assert_eq!(table.dispatch(&int(7)), 20);
        assert_eq!(table.dispatch(&int(8)), 30, "exclusive end");
        assert_eq!(table.dispatch(&int(10)), 30, "inclusive end");
        assert_eq!(table.dispatch(&int(11)), 99);
    }

    /// Hashing one would panic, so it must never reach the hasher — and it
    /// must still be able to reach the default.
    #[test]
    fn a_non_hashable_subject_falls_through_rather_than_panicking() {
        let one = int(1);
        let table = table(&[(&one, 10)], Vec::new(), 99);

        // A bare function pointer *is* hashable; only one carrying an
        // environment is not. A host type is the reliable case.
        let non_hashable = Dynamic::from(Opaque);
        assert!(
            !non_hashable.is_hashable(),
            "this test needs an non-hashable value",
        );
        assert_eq!(table.dispatch(&non_hashable), 99);
    }

    #[test]
    fn a_non_hashable_case_has_no_hash_to_key_on() {
        assert_eq!(case_hash(&Dynamic::from(Opaque)), None);
        assert!(case_hash(&int(1)).is_some());
    }

    /// What both spellings answer for `hash`, once it is established that they
    /// answer the same thing.
    ///
    /// Called rather than `case_target` so that a test says which lookups it
    /// covers instead of asking a process-global switch, which another test
    /// running beside it could be reading at the same time.
    #[track_caller]
    fn lookup(table: &Switch, hash: u64) -> Option<u32> {
        let indexed = table.case_target_indexed(hash);
        assert_eq!(
            indexed,
            table.case_target_bisected(hash),
            "the two lookups disagree about hash {hash}",
        );
        indexed
    }

    /// `switch_index` picks between two spellings of one question, so the whole
    /// claim it rests on is that they answer alike. Every size from empty up,
    /// each entry looked up and one hash no entry holds.
    #[test]
    fn both_lookups_answer_alike_at_every_size() {
        for count in 0..40 {
            let cases = spread(count);
            let table = Switch::new(cases.clone(), Vec::new(), 7);

            for case in &cases {
                assert_eq!(
                    lookup(&table, case.hash),
                    Some(case.target),
                    "{count} cases, hash {}",
                    case.hash,
                );
            }
            // One no case can hold: the spread never lands on it.
            assert_eq!(lookup(&table, 1), None, "{count} cases, absent hash");
        }
    }

    /// The same, over tables the spread does not produce: hashes drawn from a
    /// space narrow enough that entries collide in the index and repeat in
    /// `cases`, which is where two lookups are most likely to part company.
    #[test]
    fn both_lookups_answer_alike_on_colliding_tables() {
        // xorshift, so a failure names a table that can be built again.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for round in 0..2000 {
            let count = (next() % 24) as usize;
            let space = if round % 3 == 0 { 8 } else { u64::MAX };
            let mut cases: Vec<SwitchCase> = (0..count)
                .map(|i| SwitchCase {
                    hash: next() % space,
                    target: 1000 + i as u32,
                })
                .collect();
            cases.sort_by_key(|case| case.hash);
            let table = Switch::new(cases.clone(), Vec::new(), 7);

            for case in &cases {
                // Not `Some(case.target)`: a repeated hash answers with the
                // first entry holding it, which need not be this one.
                let first = cases
                    .iter()
                    .find(|other| other.hash == case.hash)
                    .expect("the hash is in the table");
                assert_eq!(
                    lookup(&table, case.hash),
                    Some(first.target),
                    "round {round}, {count} cases",
                );
            }
            for _ in 0..4 {
                lookup(&table, next() % space);
            }
        }
    }

    /// Every table with a case gets an index, and it is built with room to
    /// spare: that is what keeps a walk short, and it is what leaves the empty
    /// slot a walk stops at.
    #[test]
    fn a_table_with_cases_carries_an_index_with_room_in_it() {
        assert!(Switch::new(Vec::new(), Vec::new(), 7).index.is_empty());

        for count in 1..40 {
            let table = Switch::new(spread(count), Vec::new(), 7);
            assert!(table.index.len().is_power_of_two(), "{count} cases");
            assert!(
                table.index.len() >= 2 * table.cases().len(),
                "{count} cases",
            );
        }
    }

    /// A hash of zero is the value an empty slot holds, so a case carrying one
    /// is the case that would be lost to occupancy kept in the wrong field.
    #[test]
    fn a_case_hashing_to_zero_is_still_found() {
        let cases = spread(20);
        assert!(
            cases.iter().any(|case| case.hash == 0),
            "the spread starts at zero, so this table has the case",
        );
        let table = Switch::new(cases, Vec::new(), 7);

        assert_eq!(lookup(&table, 0), Some(100));
    }

    /// Rhai has already merged the arms sharing a case value, so the compiler
    /// cannot emit a repeated hash — but a corrupt artifact can be read into
    /// one, and both lookups have to answer it the way a scan of `cases`
    /// would: with the first entry.
    #[test]
    fn a_repeated_hash_answers_with_the_first_entry() {
        for extra in [0, 16] {
            let mut cases = vec![
                SwitchCase {
                    hash: 5,
                    target: 10,
                },
                SwitchCase {
                    hash: 5,
                    target: 20,
                },
            ];
            cases.extend(spread(extra));
            cases.sort_by_key(|case| case.hash);

            let table = Switch::new(cases, Vec::new(), 99);
            assert_eq!(lookup(&table, 5), Some(10), "with {extra} more cases");
        }
    }

    /// A target is written down twice — in `cases` and in the index — so a
    /// rewrite that reached only one of them would leave the two lookups
    /// answering differently, which is exactly what `retarget` exists to stop.
    #[test]
    fn retargeting_moves_both_copies_of_a_target() {
        let mut table = Switch::new(
            spread(12),
            vec![SwitchRange {
                from: 0,
                to: 4,
                inclusive: false,
                target: 3,
            }],
            7,
        );

        table
            .retarget(|target| {
                *target += 1000;
                Ok::<(), ()>(())
            })
            .expect("nothing refuses");

        for case in table.cases().to_vec() {
            assert_eq!(lookup(&table, case.hash), Some(case.target));
            assert!(case.target >= 1100, "the case list moved too");
        }
        assert_eq!(table.ranges[0].target, 1003);
        assert_eq!(table.default, 1007);
    }

    /// The probe is only worth carrying if it actually depends on the seed.
    #[test]
    fn the_probe_is_stable_within_a_process() {
        assert_eq!(probe(), probe());
        assert_ne!(probe(), 0, "a probe of zero could not be told from absent");
    }
}
