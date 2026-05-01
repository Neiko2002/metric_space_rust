//! This implementation of Richard's NN table builder

pub mod functions;
mod knn_search;
mod matrix;
mod table_initialisation_bsp;
mod updates;

use bits::container::BitsContainer;
use bits::{evp::matrix_dot, evp::similarity_as_f32, EvpBits};
use dao::{Dao, DaoMatrix};
use ndarray::parallel::prelude::*;
use ndarray::{concatenate, Array1, Array2, ArrayViewMut1, Axis};
use rand::{rng, Rng};
use rand_chacha::rand_core::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::BuildHasherDefault;
use std::io::Write;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use utils::non_nan::NonNan;
use utils::pair::Pair;
use utils::{arg_sort_big_to_small_2d, min_index_and_value, rand_perm};
use utils::{min_index_and_value_neighbourlarities, minimum_in_nality, Nality};

use crate::functions::{
    fill_false_atomic_from_selectors, fill_selected, get_1_d_slice_using_selected_u32,
    get_2_d_slice_using, get_reverse_nality_links_not_in_forward,
};
pub use functions::{get_selectors_from_flags, get_slice_using_selectors};

use crate::table_initialisation_bsp::*;

pub use crate::matrix::initialise_table_m;
pub use table_initialisation_bsp::{
    initialise_table_bsp, initialise_table_bsp_randomly_overwrite_row_0,
    initialise_table_bsp_randomly_overwrite_row_0_with_coin_toss,
    only_initialise_table_bsp_randomly,
};
use utils::address::{GlobalAddress, LocalAddress};

#[derive(Serialize, Deserialize)]
pub struct RDescent {
    pub neighbours: Array2<usize>,
    pub similarities: Array2<f32>,
}

pub trait IntoRDescent {
    fn into_rdescent(
        self: Rc<Self>,
        num_neighbours: usize,
        reverse_list_size: usize,
        delta: f64,
    ) -> RDescent;
}

pub struct RDescentWithRev {
    pub rdescent: RDescent,
    pub reverse_neighbours: Array2<usize>,
}

pub trait IntoRDescentWithRevNNs {
    fn into_rdescent_with_rev_nn(
        self: Rc<Self>,
        num_neighbours: usize,
        reverse_list_size: usize,
        delta: f64,
        nns_in_search_structure: usize,
    ) -> RDescentWithRev;
}

pub trait KnnSearch<T: Clone> {
    fn knn_search(
        &self,
        query: T,
        dao: Rc<Dao<T>>,
        num_neighbours: usize,
        distance: fn(&T, &T) -> f32,
    ) -> (usize, Vec<Pair>);
}

pub trait RevSearch<T: Clone> {
    fn rev_search(
        &self,
        query: T,
        dao: &Dao<T>,
        num_neighbours: usize,
        distance: fn(&T, &T) -> f32,
    ) -> Array1<usize>;
}

//********** Implementations of RDescent and RDescentRev **********

impl<C: BitsContainer, const W: usize> RevSearch<EvpBits<C, W>> for RDescentWithRev {
    /* The function uses NN and revNN tables to query in the manner of descent
     * We start with a rough approximation of the query by selecting eg 1000 distances
     * Then we iterate to see if any of the NNs of these NNs are closer to the query, using the NN table directly but also the reverseNN table
     */

    fn rev_search(
        &self,
        query: EvpBits<C, W>,
        dao: &Dao<EvpBits<C, W>>,
        num_neighbours: usize,
        distance: fn(&EvpBits<C, W>, &EvpBits<C, W>) -> f32,
    ) -> Array1<usize> {
        let data = dao.get_data();

        let mut q_nns: Array1<usize> =
            Array1::from_shape_fn((num_neighbours,), |_| rng().random_range(0..dao.num_data));
        let mut q_sims: Array1<f32> = Array1::from_shape_fn((num_neighbours,), |i| {
            similarity_as_f32::<C, W>(&data[q_nns[i as usize]], &query)
        });

        let query_as_array = Array1::from_elem(1, query);

        // same as in nnTableBuild, the new flags
        let mut new_flags: Array1<bool> = Array1::from_elem(q_nns.len(), true);
        let mut current_min_sim = *q_sims.last().unwrap(); // least good sim from q_sims

        // The amount of work done in the iteration
        let mut work_done = 1;

        while work_done > 0 {
            work_done = 0;

            // q_nns are the current best NNs that we know about
            // but don't re-try ones that have already been added before this loop

            let selectors = get_selectors_from_flags(&new_flags);

            let these_q_nns: Array1<usize> =
                get_slice_using_selectors(&q_nns.view(), &selectors.view());

            // set all to false; will be reset to true when overwritten with new values
            new_flags.fill(false);

            // get the friends of the new ones
            // forward_nns starts off as an X x k array if there are X these_q_nns

            // TODO what if these_q_nns is bigger than number of neighbours in the table

            let forward_nns: Array1<u32> =
                get_2_d_slice_using(&self.rdescent.neighbours.view(), &these_q_nns.view())
                    .flatten()
                    .map(|x| *x as u32);
            // .into_owned();

            // these two lines do the same for the reverse table as above for the forward table

            let reverse_nns: Array1<u32> =
                get_2_d_slice_using(&self.reverse_neighbours.view(), &these_q_nns.view())
                    .flatten()
                    .into_iter()
                    .filter_map(|x| {
                        if x == u32::MAX as usize {
                            None
                        } else {
                            Some(x as u32)
                        }
                    })
                    .collect();

            // TODO eliminate duplicates from reverse_nns - but leave for now - expensive and tricky
            let all_ids = concatenate(Axis(0), &[forward_nns.view(), reverse_nns.view()]).unwrap();

            // Remove zeros (which are not encoded as zeros but as maxints)
            let all_ids = all_ids
                .into_iter()
                .filter(|&id| id != u32::MAX)
                .collect::<Array1<u32>>();

            // get a view of the actual data values from the full data set but not the zeros.

            let nn_data: Array1<EvpBits<C, W>> =
                get_1_d_slice_using_selected_u32(&data, &all_ids.view());

            // and measure the similarity of each to the query
            // allSims is a flat vector is distances
            // ie it is a 1 x N array where N is the number of elements in allIds
            // only it isnt so needs flattening
            let all_sims = matrix_dot(nn_data.view(), query_as_array.view(), |a, b| {
                similarity_as_f32::<C, W>(a, b)
            });

            let all_sims = all_sims.flatten();

            for neighbour_index in 0..all_ids.len() {
                // this code is the same as just one of the four bits of phase 3 in the nn table build algorithm
                let this_id = all_ids[neighbour_index];
                let this_sim = all_sims[neighbour_index];
                // is the similarity of the query and thisId greater than the smallest similarity in the result set?
                // if it's not, then do nothing and carry on

                if this_sim > current_min_sim {
                    // more similar than what we have so far
                    if !q_nns.iter().any(|x| *x == this_id as usize) {
                        // check if this_id is in the result set already
                        // and it's not, so we're doing a replacement
                        // first find where the current smallest similarity is
                        let (position, _) = min_index_and_value(&q_sims.view());
                        // then replace the id in the result list with the new id, also maintaining the global q_sims list
                        q_nns[position] = this_id as usize;
                        q_sims[position] = this_sim;
                        new_flags[position] = true;

                        let (_, min) = min_index_and_value(&q_sims.view());
                        current_min_sim = min;

                        // and log that we've done some work so we don't want to stop yet
                        work_done += 1;
                    }
                }
            }
        }

        q_nns
    }
}

// TODO WHY IS THIS COMMENTED? Ben?? <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// impl IntoRDescentWithRevNNs for DaoMatrix<f32> {
//     fn into_rdescent_with_rev_nn(
//         self: Rc<Self>,
//         num_neighbours: usize,
//         reverse_list_size: usize,
//         delta: f64,
//         nns_in_search_structure: usize,
//     ) -> RDescentWithRev {
//     let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142);
//     let (mut neighbours, mut similarities) =
//         initialise_table_m(self.clone(), chunk_size, num_neighbours);
//     get_nn_table2_m(
//         self.clone(),
//         &mut neighbours,
//         &mut similarities,
//         num_neighbours,
//         delta,
//         reverse_list_size,
//     );
//     let (reverse_nns, _reverse_similarities) =
//         get_reverse_links_not_in_forward(&neighbours, &similarities, nns_in_search_structure);
//
//     let r_descent = RDescentMatrix {
//         neighbours: neighbours,
//         similarities: similarities,
//     };
//
//     RDescentMatrixWithRev {
//         rdescent: r_descent,
//         reverse_neighbours: reverse_nns,
//     }
// }
//}

const ZERO: u32 = 0;

impl<C: BitsContainer, const W: usize> IntoRDescent for Dao<EvpBits<C, W>> {
    fn into_rdescent(
        self: Rc<Self>,
        num_neighbours: usize,
        reverse_list_size: usize,
        delta: f64,
    ) -> RDescent {
        //let rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142);
        let neighbourlarities = initialise_table_bsp_randomly_overwrite_row_0(
            self.clone().num_data,
            num_neighbours,
            ZERO,
        );

        get_nn_table2_bsp(
            self.clone(),
            &neighbourlarities,
            num_neighbours,
            delta,
            reverse_list_size,
        );

        let ords = neighbourlarities.mapv(|x| x.id().as_usize());
        let dists = neighbourlarities.mapv(|x| x.sim());

        RDescent {
            neighbours: ords,
            similarities: dists,
        }
    }
}

impl<C: BitsContainer, const W: usize> IntoRDescentWithRevNNs for Dao<EvpBits<C, W>> {
    fn into_rdescent_with_rev_nn(
        self: Rc<Self>,
        num_neighbours_in_nn_table: usize,
        build_reverse_list_size: usize,
        delta: f64,
        nns_in_search_structure: usize,
    ) -> RDescentWithRev {
        // let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(324 * 142); TODO delete me

        let neighbourlarities = initialise_table_bsp_randomly_overwrite_row_0(
            self.clone().num_data,
            num_neighbours_in_nn_table,
            ZERO,
        );

        get_nn_table2_bsp(
            self.clone(),
            &neighbourlarities,
            num_neighbours_in_nn_table,
            delta,
            build_reverse_list_size,
        );

        let reverse =
            get_reverse_nality_links_not_in_forward(&neighbourlarities, nns_in_search_structure);

        let neighbours = neighbourlarities.mapv(|x| x.id().as_usize());
        let similarities = neighbourlarities.mapv(|x| x.sim());
        let reverse_ids = reverse.mapv(|x| {
            if x.is_empty() {
                u32::MAX as usize
            } else {
                x.id().as_usize()
            }
        });

        RDescentWithRev {
            rdescent: RDescent {
                neighbours,
                similarities,
            },
            reverse_neighbours: reverse_ids,
        }
    }
}

pub fn check_apply_update(
    row_id: usize,
    new_nality_addr: u32,
    new_nality_similarity: f32,
    neighbour_is_new: &Array2<AtomicBool>,
    neighbourlarities: &Array2<Nality>,
    work_done: &AtomicUsize,
) -> () {
    loop {
        // We expect the old value to be the same as the new if there is no contention.
        let (min_col_id, min_naility) = minimum_in_nality(&neighbourlarities.row(row_id));
        if new_nality_similarity > min_naility.sim() {
            if neighbourlarities
                .row(row_id)
                .iter()
                .any(|x| x.id().as_u32() == new_nality_addr)
            {
                // If we see the id we're inserting, bomb out
                return;
            }

            let min_ality_before_check = min_naility.get().load(Ordering::SeqCst); // get the current min_nality as am Atomic u64
            let new_value_to_add =
                Nality::new(new_nality_similarity, GlobalAddress::into(new_nality_addr)); // this is the new Naility to add to the row

            // And try to insert the new one if it's not been changed...
            // only succeeds if the current value if the same as min_ality_before_check
            // works on u44 hence need for neighbourlarities[[row_id,min_col_id]].get() - this get() is thread safe - really only a view.
            match neighbourlarities[[row_id, min_col_id]]
                .get()
                .compare_exchange(
                    min_ality_before_check, // the expected value to be in location
                    new_value_to_add.get().load(Ordering::SeqCst), // the new value
                    Ordering::SeqCst,       // success ordering
                    Ordering::SeqCst,       // failure ordering
                ) {
                Ok(_) => {
                    // we have done the swap and all is good.
                    neighbour_is_new[[row_id, min_col_id]].store(true, Ordering::Relaxed); // update the new flag
                    work_done.fetch_add(1, Ordering::Relaxed); // record that we have dome some work
                    return;
                }

                Err(updated_value_in_slot) => {
                    // let similarity = x as f32; // Nasty hack to get the similarity <<<<<< ?????????
                    let updated_slot_similarity = Nality::new_from_u64(updated_value_in_slot).sim();

                    // The least similar thing is now something better than the update we are applying... bomb out
                    if updated_slot_similarity >= new_nality_similarity {
                        return;
                    }
                }
            }
        } else {
            return; // If the update sim is smaller than the current min sim, return...
        }
    }
}

pub fn get_nn_table_up_to_phase2_bsp<C: BitsContainer, const W: usize>(
    dao: Rc<Dao<EvpBits<C, W>>>,
    neighbourlarities: &Array2<Nality>,
    num_neighbours: usize,
    delta: f64,
    reverse_list_size: usize,
) {
    let num_data = dao.num_data;
    let data = dao.get_data();
    let mut neighbour_is_new =
        Array2::from_shape_fn((num_data, num_neighbours), |_| AtomicBool::new(true));
    let mut work_done: AtomicUsize = AtomicUsize::new(num_data);

    // Phase 1
    let now = Instant::now();
    let mut new: Array2<Nality> =
        Array2::from_elem((num_data, num_neighbours), Nality::new_empty());
    let mut old: Array2<Nality> =
        Array2::from_elem((num_data, num_neighbours), Nality::new_empty());

    for row in 0..num_data {
        let row_flags = neighbour_is_new.row_mut(row);
        let new_indices = row_flags
            .iter()
            .enumerate()
            .filter_map(|(index, flag)| if flag.load(Ordering::Relaxed) { Some(index) } else { None })
            .collect::<Array1<usize>>();
        let old_indices = row_flags
            .iter()
            .enumerate()
            .filter_map(|(index, flag)| if !flag.load(Ordering::Relaxed) { Some(index) } else { None })
            .collect::<Array1<usize>>();
        let sampled = rand_perm(new_indices.len(), (new_indices.len() as f64).round() as u64 as usize);
        let mut new_row_view: ArrayViewMut1<Nality> = new.row_mut(row);
        let mut old_row_view: ArrayViewMut1<Nality> = old.row_mut(row);
        let mut neighbour_row_view: ArrayViewMut1<AtomicBool> = neighbour_is_new.row_mut(row);
        fill_selected(&mut new_row_view, &neighbourlarities.row(row), &sampled.view());
        fill_selected(&mut old_row_view, &neighbourlarities.row(row), &old_indices.view());
        fill_false_atomic_from_selectors(&mut neighbour_row_view, &sampled.view())
    }
    log::debug!("Phase 1 (debug): {} ms", ((Instant::now() - now).as_millis() as f64));

    // Phase 2
    let now = Instant::now();
    let mut reverse: Array2<Nality> =
        Array2::from_elem((num_data, reverse_list_size), Nality::new_empty());
    let mut reverse_count = Array1::from_elem(num_data, 0);

    for row in 0..num_data {
        let this_row_neighbourlarities = &neighbourlarities.row(row);
        for id in 0..num_neighbours {
            let this_id = this_row_neighbourlarities[id].id().as_usize();
            let local_sim = this_row_neighbourlarities[id].sim();
            let new_forward_links = new.row(this_id);
            let forward_links_dont_contain_this = !new_forward_links.iter().any(|x| x.id().as_usize() == row);
            if forward_links_dont_contain_this {
                if reverse_count[this_id] < reverse_list_size {
                    reverse[[this_id, reverse_count[this_id]]] = Nality::new(
                        local_sim,
                        GlobalAddress::into(row.try_into().unwrap_or_else(|_| panic!("Cannot convert usize to u32"))),
                    );
                    reverse_count[this_id] = reverse_count[this_id] + 1;
                } else {
                    let (position, value) = min_index_and_value_neighbourlarities(&reverse.row(this_id));
                    if value.sim() < local_sim {
                        reverse[[this_id, position as usize]] = Nality::new(
                            local_sim,
                            GlobalAddress::into(row.try_into().unwrap_or_else(|_| panic!("Cannot convert usize to u32"))),
                        );
                    }
                }
            }
        }
    }
    log::debug!("Phase 2 (debug): {} ms", ((Instant::now() - now).as_millis() as f64));
}

pub fn get_nn_table2_bsp<C: BitsContainer, const W: usize>(

    dao: Rc<Dao<EvpBits<C, W>>>,
    neighbourlarities: &Array2<Nality>,
    num_neighbours: usize,
    delta: f64,
    reverse_list_size: usize,
) {
    log::warn!("Neighbourities length: {:?}", neighbourlarities.shape(),);

    let start_time = Instant::now();

    let num_data = dao.num_data;
    let data = dao.get_data();

    // Matlab lines refer to richard_build.txt file in the matlab dir

    let mut iterations = 0;

    let mut neighbour_is_new =
        Array2::from_shape_fn((num_data, num_neighbours), |_| AtomicBool::new(true));

    let mut work_done: AtomicUsize = AtomicUsize::new(num_data); // a count of the number of times a similarity minimum of row has changed - measure of flux

    while work_done.load(std::sync::atomic::Ordering::SeqCst) > ((num_data as f64) * delta) as usize
    {
        // Matlab line 61
        // condition is fraction of lines whose min similarity has changed when this gets low - no much work done then stop.
        iterations += 1;

        log::debug!(
            "iterating: c: {} num_data: {} iters: {}",
            work_done.load(std::sync::atomic::Ordering::SeqCst),
            num_data,
            iterations
        );

        // phase 1

        let now = Instant::now();

        let mut new: Array2<Nality> =
            Array2::from_elem((num_data, num_neighbours), Nality::new_empty()); // Matlab line 65
        let mut old: Array2<Nality> =
            Array2::from_elem((num_data, num_neighbours), Nality::new_empty());

        // initialise old and new inline

        for row in 0..num_data {
            // in Matlab line 74
            let row_flags = neighbour_is_new.row_mut(row); // Matlab line 74

            // new_indices are the indices in this row whose flag is set to true (columns)

            let new_indices = row_flags // Matlab line 76
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| {
                    if flag.load(Ordering::Relaxed) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect::<Array1<usize>>();

            // old_indices are the indices in this row whose flag is set to false (intially there are none of these).

            let old_indices = row_flags // Matlab line 77
                .iter()
                .enumerate()
                .filter_map(|(index, flag)| {
                    if !flag.load(Ordering::Relaxed) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect::<Array1<usize>>();

            // random data ids from whole data set
            // in matlab p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n

            let sampled = rand_perm(
                new_indices.len(),
                (new_indices.len() as f64).round() as u64 as usize,
            );

            // sampled are random indices from new_indices

            let mut new_row_view: ArrayViewMut1<Nality> = new.row_mut(row);
            let mut old_row_view: ArrayViewMut1<Nality> = old.row_mut(row);
            let mut neighbour_row_view: ArrayViewMut1<AtomicBool> = neighbour_is_new.row_mut(row);

            fill_selected(
                &mut new_row_view,
                &neighbourlarities.row(row),
                &sampled.view(),
            ); // Matlab line 79

            fill_selected(
                &mut old_row_view,
                &neighbourlarities.row(row),
                &old_indices.view(),
            );

            fill_false_atomic_from_selectors(&mut neighbour_row_view, &sampled.view())
        }

        let after = Instant::now();
        log::debug!("Phase 1: {} ms", ((after - now).as_millis() as f64));

        // phase 2  Matlab line 88

        let now = Instant::now();

        // initialise old' and new'  Matlab line 90

        // // the reverse NN table  Matlab line 91
        // let mut reverse: Array2<usize> = Array2::from_elem((num_data, reverse_list_size), 0);
        // // all the distances from reverse NN table.
        // let mut reverse_sims: Array2<f32> =
        //     Array2::from_elem((num_data, reverse_list_size), -1.0f32);

        let mut reverse: Array2<Nality> =
            Array2::from_elem((num_data, reverse_list_size), Nality::new_empty());
        // reverse_ptr - how many reverse pointers for each entry in the dataset
        // FERDIA: brings us to 18GB here
        let mut reverse_count = Array1::from_elem(num_data, 0);

        // loop over all current entries in neighbours; add that entry to each row in the
        // reverse list if that id is in the forward NNs
        // there is a limit to the number of reverse ids we will store, as these
        // are in a zipf distribution, so we will add the most similar only

        for row in 0..num_data {
            // Matlab line 97
            // all_ids are the forward links in the current id's row
            let this_row_neighbourlarities = &neighbourlarities.row(row); // Matlab line 98
                                                                          // so for each one of these (there are k...):
            for id in 0..num_neighbours {
                // Matlab line 99 (updated)
                // get the id
                let this_id = this_row_neighbourlarities[id].id().as_usize();
                // and how similar it is to the current id
                let local_sim = this_row_neighbourlarities[id].sim();

                // newForwardLinks = new(thisId,:);
                let new_forward_links = new.row(this_id);

                // forwardLinksDontContainThis = sum(newForwardLinks == i_phase2) == 0;
                let forward_links_dont_contain_this =
                    !new_forward_links.iter().any(|x| x.id().as_usize() == row);

                // if the reverse list isn't full, we will just add this one
                // this adds to a priority queue and keeps track of max

                // We are trying to find a set of reverse near neighbours with the
                // biggest similarity of size reverse_list_size.
                // first find all the forward links containing the row

                if forward_links_dont_contain_this {
                    // if the reverse list isn't full, we will just add this one
                    // this adds to a priority queue and keeps track of max

                    // We are trying to find a set of reverse near neighbours with the
                    // biggest similarity of size reverse_list_size.
                    // first find all the forward links containing the row

                    if reverse_count[this_id] < reverse_list_size {
                        // if the list is not full
                        // update the reverse pointer list and the similarities

                        reverse[[this_id, reverse_count[this_id]]] = Nality::new(
                            local_sim,
                            GlobalAddress::into(
                                row.try_into()
                                    .unwrap_or_else(|_| panic!("Cannot convert usize to u32")),
                            ),
                        );
                        reverse_count[this_id] = reverse_count[this_id] + 1; // increment the count
                    } else {
                        // the list is full - so no need to do anything with counts
                        // but it is, so we will only add it if it's more similar than another one already there

                        let (position, value) =
                            min_index_and_value_neighbourlarities(&reverse.row(this_id)); // Matlab line 109
                        let value = value.sim();

                        if value < local_sim {
                            // Matlab line 110  if the value in reverse_sims is less similar we over write
                            reverse[[this_id, position as usize]] = Nality::new(
                                local_sim,
                                GlobalAddress::into(
                                    row.try_into()
                                        .unwrap_or_else(|_| panic!("Cannot convert usize to u32")),
                                ),
                            );
                        }
                    }
                }
            }
        }

        let after = Instant::now();
        log::debug!("Phase 2: {} ms", ((after - now).as_millis() as f64));

        // phase 3

        let now = Instant::now();

        work_done = AtomicUsize::new(0);

        // let updates = Updates::new(num_data); // TODO delete

        // let mut mutexes = vec![];
        // for i in 0..num_data {
        //     mutexes.push(Mutex::new(()));
        // }

        old.axis_iter_mut(Axis(0)) // Get mutable rows (disjoint slices)
            .enumerate()
            .zip(new.axis_iter_mut(Axis(0)))
            .par_bridge()
            .map(|((row, old_row), new_row)| {
                let binding = reverse
                    .row(row);


                let new_row_union: Array1<usize> = if new_row.len() == 0 {
                    // Matlab line 130
                    Array1::from(vec![])
                } else {
                    new_row
                        .iter()
                        .filter_map(|x| { if x.is_empty() {None} else { Some( x.id().as_usize() ) } } ) //<<<<<<<<< only take real values
                        .chain(binding
                                .iter()
                                .filter(|&x| !x.is_empty())
                                .map(|x| x.id().as_usize()))
                        .collect::<Array1<usize>>()
                };

                // index the data using the rows indicated in old_row
                let old_data = get_slice_using_selectors(
                    &data,
                    &old_row
                        .iter()
                        .map(|x| { x.id().as_usize() } )
                        .collect::<Array1<_>>().view(),
                ); // Matlab line 136

                let new_data = get_slice_using_selectors(
                    &data,
                    &new_row
                        .iter()
                        .map(|x| x.id().as_usize() )
                        .collect::<Array1<_>>().view(),
                ); // Matlab line 137

                let new_union_data =
                    get_slice_using_selectors(&data, &new_row_union.view()); // Matlab line 137

                let new_new_sims =
                    matrix_dot(new_union_data.view(), new_union_data.view(), |a, b| {
                        similarity_as_f32(a, b)
                    });

                (

                    new_row,
                    old_row,
                    new_row_union,
                    new_new_sims,
                    new_data,
                    old_data,
                )
            })
            .for_each(
                |(new_row,
                     old_row,
                     new_row_union,
                     new_new_sims,
                     new_data,
                     old_data)| {
                    // Two for loops for the two distance tables (similarities and new_old_sims) for each pair of elements in the newNew list, their original ids
                    // First iterate over new_new_sims.. upper triangular (since distance table)

                    if new_row_union.len() >= 2 {
                        // must be at least 2 elements in the array because we are doing pair-wise comparisons.

                        for new_ind1 in 0..new_row_union.len() - 1 {
                            // Matlab line 144 (-1 since don't want the diagonal)
                            let u1_id = *new_row_union.get(new_ind1).unwrap_or_else(|| panic!("Illegal index of new_row_union at {new_ind1} length is: {}", new_row_union.len()));
                            let u1_id_as_u32 = u1_id.try_into()
                                .unwrap_or_else(|_| panic!("Cannot convert u1_id to u32"));

                            for new_ind2 in new_ind1 + 1..new_row_union.len() {
                                // Matlab line 147
                                let u2_id = *new_row_union.get(new_ind2).unwrap_or_else(|| panic!("Illegal index of new_row_union at {new_ind2} length is: {}", new_row_union.len()));
                                let u2_id_as_u32 = u2_id.try_into()
                                    .unwrap_or_else(|_| panic!("Cannot convert u2_id to u32"));
                                // then get their similarity from the matrix
                                let this_sim = *new_new_sims.get((new_ind1, new_ind2)).unwrap_or_else(|| panic!("Illegal index of new_new_sims at {new_ind1},{new_ind2} Shape is: {:?}", new_new_sims.shape()));
                                // is the current similarity greater than the biggest distance
                                // in the row for u1_id? if it's not, then do nothing

                                check_apply_update(
                                    u1_id,
                                    u2_id_as_u32, // &Nality::new(this_sim, ),
                                    this_sim,
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );
                                check_apply_update(
                                    u2_id,
                                    u1_id_as_u32,  // &Nality::new(this_sim, ),
                                    this_sim,
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );
                            } // Matlab line 175
                        }

                        // now do the news vs the olds, no reverse links

                        let new_old_sims =
                            matrix_dot(new_data.view(),
                                                old_data.view(),
                                                |a, b| { similarity_as_f32(a, b) }
                            );

                        // and do the same for each pair of elements in the new_row/old_row

                        for new_ind1 in 0..new_row.len() {
                            // Matlab line 183  // rectangular matrix - need to look at all

                            let u1 = new_row.get(new_ind1).unwrap_or_else(|| panic!("Illegal index of new_row at {new_ind1} length is: {}", new_row.len()));
                            for new_ind2 in 0..old_row.len() {
                                let u2 = old_row.get(new_ind2).unwrap(); // Matlab line 186

                                // then get their distance from the matrix

                                let this_sim = *new_old_sims.get((new_ind1, new_ind2)).unwrap_or_else(|| panic!("Illegal index of new_old_sims at {new_ind1},{new_ind2} Shape is: {:?}", new_old_sims.shape()));

                                check_apply_update(
                                    u1.id().as_usize(),   // the new row
                                    u2.id().as_u32(),   // <<<<<<<<<< the new index to be added to row
                                    this_sim,           // with this similarity
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );

                                check_apply_update(
                                    u2.id().as_usize(),
                                    u1.id().as_u32(),
                                    this_sim,
                                    &neighbour_is_new,
                                    neighbourlarities,
                                    &work_done,
                                );
                            }
                        }
                    }
                },
            );

        let after = Instant::now();
        log::debug!("Phase 3: {} ms", ((after - now).as_millis() as f64));
    }

    log::debug!(
        "Final iteration: c: {} iters: {}",
        work_done.load(std::sync::atomic::Ordering::SeqCst),
        iterations
    );

    let final_time = Instant::now();
    log::trace!(
        "Overall time 3: {} ms",
        ((final_time - start_time).as_millis() as f64)
    );
}
