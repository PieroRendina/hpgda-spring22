// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <sstream>
#include "personalized_pagerank.cuh"
#include <list>
#include <vector>
namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

//////////////////////////////
//////////////////////////////
#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

// Write GPU kernel here!
__global__ void modify_device_array_value(double *device_array, int index, double value)
{
    // residues[personalization_vertex] = 1.0;
    device_array[index] = value;
}

__global__ void update_pi0_and_r(int *frontier_d, double alpha, double *pi0_d, double *r_d, int dim_frontier)
{
    // compute the index of the vertex in the frontier
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim_frontier; i += blockDim.x * gridDim.x)
    {
        pi0_d[frontier_d[i]] += alpha * r_d[frontier_d[i]];
        r_d[frontier_d[i]] = 0.0;
    }
}

//  TODO change this function because it doesn't work
__global__ void compute_new_frontier(double *r_d, double rmax, bool *flags_d, int *outdegrees, double alpha, int *out_neighbors, int tot_neighbors, int dim_frontier)
{
    /*for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < tot_neighbors ; j += blockDim.x * gridDim.x) {
        if(outdegrees[j] > 0){
            r_d[out_neighbors[j]] += (1 - alpha)*r_d[out_neighbors[j]]/outdegrees[j];
            if(r_d[out_neighbors[j]]/outdegrees[j] > rmax && flags_d[out_neighbors[j]] != true) {
                //thread syncing before updating the flags
                __syncthreads();
                flags_d[out_neighbors[j]] = true;
          }
        }else{
            r_d[out_neighbors[j]] = 0;
        }
    }*/
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < dim_frontier; j += blockDim.x)
    {
        if (outdegrees[j] > 0)
        {
            int neighbor_idx = 0;
            if (j - 1 >= 0)
            {
                neighbor_idx += outdegrees[j - 1];
            }
            int idx = neighbor_idx;
            while (idx < neighbor_idx + outdegrees[j])
            {
                r_d[out_neighbors[idx]] += (1 - alpha) * r_d[out_neighbors[j]] / outdegrees[j];
                //printf("residue considered = %lf\n ", r_d[out_neighbors[idx]]);
                if (r_d[out_neighbors[idx]] > rmax && flags_d[out_neighbors[idx]] != true)
                {
                    __syncthreads();
                    flags_d[out_neighbors[idx]] = true;
                }
                idx++;
            }
        }
        else
        {
            r_d[out_neighbors[j]] = 0;
        }
    }
}

//////////////////////////////
//////////////////////////////

void PersonalizedPageRank::update_frontiers()
{

    /* --- Compute the number of neighbors and drop the nodes from the frontier --- */
    // allocate the vector to store the degree of each node in the frontier
    int tot_neighbors = 0;
    int *outdegrees;
    /* one outdegree for each member of the frontier */
    err = cudaMallocManaged(&outdegrees, sizeof(int) * dim_frontier);
    printf("\nDim frontier = %d\n", dim_frontier);
    for (int i = 0; i < dim_frontier; i++)
    {
        printf("node in the frontier = %d\n", frontier[i]);
        int start_idx = neighbor_start_idx[frontier[i]];
        int end_idx = neighbor_start_idx[frontier[i] + 1];
        // outdegree computation
        outdegrees[i] = end_idx - start_idx;
        tot_neighbors += outdegrees[i];
        // the node is dropped from the frontier
        flags[frontier[i]] = false;
    }
    /* --- Add the neighbours to be considered in the vector out_neighbors --- */
    int *out_neighbors;
    /* all the neighbors to be considered */
    err = cudaMallocManaged(&out_neighbors, sizeof(int) * tot_neighbors);
    int counter = 0;
    for (int i = 0; i < dim_frontier; i++)
    {
        int start_idx = neighbor_start_idx[frontier[i]];
        int end_idx = neighbor_start_idx[frontier[i] + 1];
        for (int j = start_idx; j < end_idx; j++)
        {
            out_neighbors[counter] = neighbors[j];
            counter += 1;
        }
    }
    /* --- Update of the frontier --- */
    // int new_frontier_dim = 3*dim_frontier;
    // int new_frontier_dim = 30 * dim_frontier;
    // int *new_frontier;
    /* --- For simplicity I allocate a frontier that is long the number of vertex --- */
    err = cudaMallocManaged(&new_frontier, sizeof(int)*V);
    //int *new_frontier = (int *)malloc(new_frontier_dim * sizeof(int));

    /* For each out_neighbor updates the residue and checks if it has to be added to the frontier */
    // compute_new_frontier<<<ceil(tot_neighbors/1024)+1, ceil(tot_neighbors/ceil(tot_neighbors/1024))+1>>>(residues_d, rmax, flags_d, outdegrees, alpha, out_neighbors, tot_neighbors);

    int n_blocks = ceil(tot_neighbors/1024) +1;
    int n_threads = ceil(tot_neighbors/n_blocks) +1;
    cudaMemcpy(flags_d, flags, sizeof(bool) * V, cudaMemcpyHostToDevice);
    compute_new_frontier<<<1, 1>>>(residues_d, rmax, flags_d, outdegrees, alpha, out_neighbors, tot_neighbors, dim_frontier);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(flags, flags_d, sizeof(bool) * V, cudaMemcpyDeviceToHost);
    
    int idx_frontier = 0;
    for (int i = 0; i < V; i++)
    {
        if (flags[i] == true)
        {
          /*
          if(idx_frontier >= new_frontier_dim){
            new_frontier_dim = 3 * new_frontier_dim;
            new_frontier = (int *)realloc(new_frontier, new_frontier_dim * sizeof(int));
            printf("--- realloc done --- \n");
          }*/
            new_frontier[idx_frontier] = i;
            idx_frontier++;
        }
    }

    /* -- Check the flag -- */

    // cudaFree(frontier);
    // frontier = new_frontier;
    new_dim_frontier = idx_frontier;
    std::cout << "----- Updated frontier -----\n";
    for (int i = 0; i < new_dim_frontier; i++)
    {
        std::cout << new_frontier[i] << " ";
    }
}

void PersonalizedPageRank::initialize_csr()
{
    verteces = (int *)malloc(V * sizeof(int));
    // allocate a vector containing the index of the starting neighbor
    neighbor_start_idx = (int *)malloc((V + 1) * sizeof(int));
    neighbors = (int *)malloc(E * sizeof(int));

    int curr_vertex = 0;
    int curr_neighbor = 0;
    int curr_neighbor_start_idx = 1;
    neighbor_start_idx[0] = 0;

    for (int i = 0; i < V; i++)
    {
        verteces[curr_vertex] = i;
        curr_vertex++;
        for (int j = 0; j < E; j++)
        {
            if (y[j] == i)
            {
                neighbors[curr_neighbor] = x[j];
                curr_neighbor++;
            }
        }
        neighbor_start_idx[curr_neighbor_start_idx] = curr_neighbor;
        curr_neighbor_start_idx++;
    }

    std::cout << "----- Verteces -----\n";
    for (int i = 0; i < V; i++)
    {
        std::cout << verteces[i] << " ";
    }

    std::cout << "\n----- Neighbours -----\n";
    for (int i = 0; i < V + 1; i++)
    {
        std::cout << neighbor_start_idx[i] << " ";
    }
    std::cout << "\n----- Neighbours indeces -----\n";
    for (int i = 0; i < E; i++)
    {
        std::cout << neighbors[i] << " ";
    }
}

// CPU Utility functions;

// Read the input graph and initialize it;
void PersonalizedPageRank::initialize_graph()
{
    // Read the graph from an MTX file;
    int num_rows = 0;
    int num_columns = 0;
    read_mtx(graph_file_path.c_str(), &x, &y, &val,
             &num_rows, &num_columns, &E, // Store the number of vertices (row and columns must be the same value), and edges;
             true,                        // If true, read edges TRANSPOSED, i.e. edge (2, 3) is loaded as (3, 2). We set this true as it simplifies the PPR computation;
             false,                       // If true, read the third column of the matrix file. If false, set all values to 1 (this is what you want when reading a graph topology);
             debug,
             false, // MTX files use indices starting from 1. If for whatever reason your MTX files uses indices that start from 0, set zero_indexed_file=true;
             true   // If true, sort the edges in (x, y) order. If you have a sorted MTX file, turn this to false to make loading faster;
    );
    if (num_rows != num_columns)
    {
        if (debug)
            std::cout << "error, the matrix is not squared, rows=" << num_rows << ", columns=" << num_columns << std::endl;
        exit(-1);
    }
    else
    {
        V = num_rows;
    }
    if (debug)
        std::cout << "loaded graph, |V|=" << V << ", |E|=" << E << std::endl;

    // Compute the dangling vector. A vertex is not dangling if it has at least 1 outgoing edge;
    dangling.resize(V);
    std::fill(dangling.begin(), dangling.end(), 1); // Initially assume all vertices to be dangling;
    for (int i = 0; i < E; i++)
    {
        // Ignore self-loops, a vertex is still dangling if it has only self-loops;
        if (x[i] != y[i])
            dangling[y[i]] = 0;
    }
    // Initialize the CPU PageRank vector;
    pr.resize(V);
    pr_golden.resize(V);
    // Initialize the value vector of the graph (1 / outdegree of each vertex).
    // Count how many edges start in each vertex (here, the source vertex is y as the matrix is transposed);
    int *outdegree = (int *)calloc(V, sizeof(int));
    for (int i = 0; i < E; i++)
    {
        outdegree[y[i]]++;
    }
    // Divide each edge value by the outdegree of the source vertex;
    for (int i = 0; i < E; i++)
    {
        val[i] = 1.0 / outdegree[y[i]];
    }
    free(outdegree);
}

//////////////////////////////
//////////////////////////////

// Allocate data on the CPU and GPU;
void PersonalizedPageRank::alloc()
{
    // Load the input graph and preprocess it;
    initialize_graph();
    initialize_csr();

    // allocate the mask to store the status of the nodes in the frontier (all false by default)
    flags = (bool *)calloc(V, sizeof(bool));
    // at the beginning the frontier contains just the personalization vertex
    // frontier = (int *)malloc(sizeof(int));
    err = cudaMallocManaged(&frontier, sizeof(int));

    // Allocate any GPU data here;
    // TODO!
    err = cudaMalloc(&verteces_d, sizeof(int) * V);
    err = cudaMalloc(&neighbor_start_idx_d, sizeof(int) * (V + 1));
    err = cudaMalloc(&neighbors_d, sizeof(int) * E);
    err = cudaMalloc(&pi0_d, sizeof(double) * V);
    err = cudaMalloc(&residues_d, sizeof(double) * V);
    err = cudaMalloc(&flags_d, sizeof(bool) * V);
    // err = cudaMalloc(&frontier_d, sizeof(int));
    //  some variables may be missing

    /*double * personal_x = (double*)malloc(sizeof(double)*V);
    cudaMemcpy(personal_x, residues_d, sizeof(double)*V, cudaMemcpyDeviceToHost);
    std::cout << "\nValue of x = " << personal_x[0];*/
}

// Initialize data;
void PersonalizedPageRank::init()
{
    // Do any additional CPU or GPU setup here;
    // TODO!

    // Compute Rmax
    threshold = 1.0 / V; // should be O(1/n) but i don't know yet which is the best value
    rmax = (convergence_threshold / sqrt(E)) * sqrt(threshold / (((2.0 * convergence_threshold / 3.0) + 2.0) * (log(2.0 / failure_probability))));
    std::cout << "rmax = " << rmax << '\n'; // It seems really small
    dim_frontier = 1;
    frontier[0] = personalization_vertex;
    flags[personalization_vertex] = true;

    cudaMemset(residues_d, 0.0, sizeof(double) * V);
    cudaMemset(pi0_d, 0.0, sizeof(double) * V);

    cudaMemcpy(verteces_d, verteces, sizeof(int) * V, cudaMemcpyHostToDevice);
    cudaMemcpy(neighbor_start_idx_d, neighbor_start_idx, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(neighbors_d, neighbors, sizeof(int) * E, cudaMemcpyHostToDevice);
    cudaMemcpy(flags_d, flags, sizeof(bool) * V, cudaMemcpyHostToDevice);

    modify_device_array_value<<<1, 1>>>(residues_d, personalization_vertex, 1.0);
    CHECK(cudaDeviceSynchronize());
}

// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset()
{
    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr.begin(), pr.end(), 1.0 / V);
    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V;
    if (debug)
        std::cout << "personalization vertex=" << personalization_vertex << std::endl;

    // Do any GPU reset here, and also transfer data to the GPU;
    // TODO!
}

void PersonalizedPageRank::execute(int iter)
{
    // Do the GPU computation here, and also transfer results to the CPU;
    // TODO! (and save the GPU PPR values into the "pr" array)
    /*while(frontier.size() > 0) {

    }*/
    // compute the dimension of the frontier
    dim_frontier = 1;
    //int dim_frontier_old; // used to update pi0 and r
  
    while(dim_frontier > 0){
      //int n_blocks = ceil(dim_frontier / 1024) + 1;
      //int n_threads = ceil(dim_frontier / n_blocks) + 1;
      //CHECK(cudaDeviceSynchronize());
      //dim_frontier_old = dim_frontier;
      update_frontiers();
      CHECK(cudaDeviceSynchronize());
      update_pi0_and_r<<<1,1>>>(frontier, alpha, pi0_d, residues_d, dim_frontier);
      CHECK(cudaDeviceSynchronize());
      frontier = new_frontier;
      dim_frontier = new_dim_frontier;
    }
    pi0 = (double*)malloc(sizeof(double)*V);
    cudaMemcpy(pi0, pi0_d, sizeof(double)*V, cudaMemcpyDeviceToHost);
    printf("\n--- Estimated pi ---\n");
    for(int i = 0; i < V; i++) {
      //printf("pi_(%d) = %lf\n", i, pi0[i]);
      pr[i] = pi0[i];
    }

}

void PersonalizedPageRank::cpu_validation(int iter)
{
    // Reset the CPU PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr_golden.begin(), pr_golden.end(), 1.0 / V);

    // Do Personalized PageRank on CPU;
    auto start_tmp = clock_type::now();
    personalized_pagerank_cpu(x.data(), y.data(), val.data(), V, E, pr_golden.data(), dangling.data(), personalization_vertex, alpha, 1e-6, 100);
    auto end_tmp = clock_type::now();
    auto exec_time = chrono::duration_cast<chrono::microseconds>(end_tmp - start_tmp).count();
    std::cout << "exec time CPU=" << double(exec_time) / 1000 << " ms" << std::endl;

    // Obtain the vertices with highest PPR value;

    std::vector<std::pair<int, double>> sorted_pr_tuples = sort_pr(pr.data(), V);
    std::vector<std::pair<int, double>> sorted_pr_golden_tuples = sort_pr(pr_golden.data(), V);

    // Check how many of the correct top-20 PPR vertices are retrieved by the GPU;
    std::unordered_set<int> top_pr_indices;
    std::unordered_set<int> top_pr_golden_indices;
    int old_precision = std::cout.precision();
    std::cout.precision(4);
    int topk = std::min(V, topk_vertices);
    for (int i = 0; i < topk; i++)
    {
        int pr_id_gpu = sorted_pr_tuples[i].first;
        int pr_id_cpu = sorted_pr_golden_tuples[i].first;
        top_pr_indices.insert(pr_id_gpu);
        top_pr_golden_indices.insert(pr_id_cpu);
        if (debug)
        {
            double pr_val_gpu = sorted_pr_tuples[i].second;
            double pr_val_cpu = sorted_pr_golden_tuples[i].second;
            if (pr_id_gpu != pr_id_cpu)
            {
                std::cout << "* error in rank! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            }
            else if (std::abs(sorted_pr_tuples[i].second - sorted_pr_golden_tuples[i].second) > 1e-6)
            {
                std::cout << "* error in value! (" << i << ") correct=" << pr_id_cpu << " (val=" << pr_val_cpu << "), found=" << pr_id_gpu << " (val=" << pr_val_gpu << ")" << std::endl;
            }
        }
    }
    std::cout.precision(old_precision);
    // Set intersection to find correctly retrieved vertices;
    std::vector<int> correctly_retrieved_vertices;
    set_intersection(top_pr_indices.begin(), top_pr_indices.end(), top_pr_golden_indices.begin(), top_pr_golden_indices.end(), std::back_inserter(correctly_retrieved_vertices));
    precision = double(correctly_retrieved_vertices.size()) / topk;
    if (debug)
        std::cout << "correctly retrived top-" << topk << " vertices=" << correctly_retrieved_vertices.size() << " (" << 100 * precision << "%)" << std::endl;
}

std::string PersonalizedPageRank::print_result(bool short_form)
{
    if (short_form)
    {
        return std::to_string(precision);
    }
    else
    {
        // Print the first few PageRank values (not sorted);
        std::ostringstream out;
        out.precision(3);
        out << "[";
        for (int i = 0; i < std::min(20, V); i++)
        {
            out << pr[i] << ", ";
        }
        out << "...]";
        return out.str();
    }
}

void PersonalizedPageRank::clean()
{
    // Delete any GPU data or additional CPU data;
    // TODO!
}
