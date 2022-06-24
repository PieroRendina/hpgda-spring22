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
#include <curand.h>
#include <curand_kernel.h>
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
/* Modify an array value on the GPU */
__global__ void modify_device_array_value(double *device_array, int index, double value) {
    device_array[index] = value;
}

/* Update the residues and the reserves of the nodes within the frontier */
__global__ void update_pi0_and_r(int *frontier_d, double alpha, double *pi0_d, double *r_d, int dim_frontier)
{
    // compute the index of the vertex in the frontier
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < dim_frontier; i += blockDim.x * gridDim.x)
    {
        pi0_d[frontier_d[i]] += alpha * r_d[frontier_d[i]];
        r_d[frontier_d[i]] = 0.0;
    }
}

/* Update the residues of the nodes directly connected to the ones in the frontier. Update the frontier itself */
__global__ void compute_new_frontier(double *r_d, double rmax, bool *flags_d, int *outdegrees, double alpha, int *out_neighbors, int tot_neighbors, int * frontier, int dim_frontier, double * rsum_d, double * pi0_d)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < dim_frontier; j += blockDim.x * gridDim.x)
    {
        int neighbor_idx = 0;
        if (j - 1 >= 0)
        {
            neighbor_idx += outdegrees[frontier[j - 1]];
        }
        int idx = neighbor_idx;
        while (idx < neighbor_idx + outdegrees[frontier[j]])
        {
            // update of the residue
            r_d[out_neighbors[idx]] += (1 - alpha) * r_d[frontier[j]] / outdegrees[frontier[j]];
            if(outdegrees[out_neighbors[idx]] > 0) {
                // check whether to insert the node in the frontier
                if (r_d[out_neighbors[idx]]/outdegrees[out_neighbors[idx]] > rmax && flags_d[out_neighbors[idx]] != true)
                {
                    __syncthreads();
                    flags_d[out_neighbors[idx]] = true;
                }
            }
            // handle the case of a dangling node, add its residues to its reserve, it for sure doesn't enter in the frontier.
            else {
                pi0_d[out_neighbors[idx]] += alpha*r_d[out_neighbors[idx]];
                r_d[out_neighbors[idx]] = 0.0;
            }
            idx++;
        }
    }
}

/* Former implementation of random walks, parallelized w.r.t. the nodes having a non-zero residue after the Forward Push step */
__global__ void random_walks(double random_walks_factor, double w, double * pi0_d, double * residues_d, int tot_nodes, int * starting_nodes, int * outdegrees, int * neighbor_start_idx_d, double alpha, int * neighbors_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(0, i, 0, &state);
    // iterate through all the starting nodes 
    for ( ; i < tot_nodes; i += blockDim.x * gridDim.x) 
    {
        // compute the number of walks to do for the node
        double wi = ceil(outdegrees[starting_nodes[i]]*random_walks_factor);
        printf("\nnumber of walks to do = %lf at iteration %d\n", wi, i);
        printf("starting node: %d, outdegree: %d\n", starting_nodes[i], outdegrees[starting_nodes[i]]);
        for (int walks_done = 0; walks_done < wi; walks_done++)
        {
            float flip = (float)curand_uniform(&state);
            int current_node = starting_nodes[i];
            while(outdegrees[current_node] > 0 && flip < (1-alpha)) {
                int decision_interval = neighbor_start_idx_d[current_node + 1] - neighbor_start_idx_d[current_node];
                int neighbor_chosen = ceil(decision_interval*curand_uniform(&state) + neighbor_start_idx_d[current_node]);
                current_node = neighbors_d[neighbor_chosen];
                flip = curand_uniform(&state);
            }
            atomicAdd(&pi0_d[current_node], residues_d[starting_nodes[i]]/wi);
        }
    }
}

/* Device function to carry out the random walks for a given node */
__global__ void perform_random_walks(double n_walks, int starting_node, int * outdegrees, int * neighbor_start_idx_d, double alpha, double * pi0_d, double * residues_d, int * neighbors_d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // initialize the random state 
    curandState state;
    curand_init(0, i, 0, &state);
    for (; i < n_walks; i += blockDim.x * gridDim.x) {
            // flip a coin to account for the probability to jump (if the value is < 0.15 and the node has some neighbors, then jump)
            float flip = (float)curand_uniform(&state);
            int current_node = starting_node;
            while(outdegrees[current_node] > 0 && flip < (1-alpha)) {
                int decision_interval = neighbor_start_idx_d[current_node + 1] - neighbor_start_idx_d[current_node];
                // choose the neighbor index to jump to.
                int neighbor_chosen = ceil(decision_interval*curand_uniform(&state) + neighbor_start_idx_d[current_node]);
                current_node = neighbors_d[neighbor_chosen];
                flip = curand_uniform(&state);
            }
            atomicAdd(&pi0_d[current_node], residues_d[starting_node]/n_walks);
    }
}


//////////////////////////////
/* CPU function */
//////////////////////////////


/* Generate and run on the GPU the random walks for each node */
void PersonalizedPageRank::generate_random_walks(int count_positive_residues) {
    for(int i = 0; i < count_positive_residues; i++) {
        double walks_per_node = ceil(outdegrees[positive_residues[i]]*random_walks_factor);
        int n_blocks = 256;
        int n_threads = 1024;
        if(i % 500 == 0) {
            printf("Starting node = %d/%d, node = %d, number of random_walks = %lf\n", i, count_positive_residues, positive_residues_d[i], walks_per_node);
        }
        perform_random_walks<<<n_blocks, n_threads>>>(walks_per_node, positive_residues_d[i], outdegrees, neighbor_start_idx_d, alpha, pi0_d, residues_d, neighbors_d);
        CHECK(cudaDeviceSynchronize());
    }
}


/* Update the frontier at each iteration of the Forward Push stage */
void PersonalizedPageRank::update_frontiers() {

    int tot_neighbors = 0;
    // Compute the number of neighbors to check
    for (int i = 0; i < dim_frontier; i++)
    {
        tot_neighbors += outdegrees[frontier[i]];
        // the node is dropped from the frontier
        flags[frontier[i]] = false;
    }
    /* --- Add the neighbours to be considered in the vector out_neighbors --- */
    int *out_neighbors;
    /* Allocate the space for the neighbors to be considered */
    err = cudaMallocManaged(&out_neighbors, sizeof(int) * tot_neighbors);
    int counter = 0;
    /* Initialize the vector of outneighbors */
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
    /* --- For simplicity I allocate a frontier that is as large as the number of vertex --- */
    err = cudaMallocManaged(&new_frontier, sizeof(int) * V);
    int n_blocks = ceil(tot_neighbors / 1024) + 1;
    int n_threads = ceil(tot_neighbors / n_blocks) + 1;
    cudaMemcpy(flags_d, flags, sizeof(bool) * V, cudaMemcpyHostToDevice);
    /* Update the flags of the nodes */
    compute_new_frontier<<<n_blocks,n_threads>>>(residues_d, rmax, flags_d, outdegrees, alpha, out_neighbors, tot_neighbors, frontier, dim_frontier, rsum_d, pi0_d);
    CHECK(cudaDeviceSynchronize());
    cudaMemcpy(flags, flags_d, sizeof(bool) * V, cudaMemcpyDeviceToHost);

    /* Initialize the new frontier */
    int idx_frontier = 0;
    for (int i = 0; i < V; i++)
    {
        if (flags[i] == true)
        {
            new_frontier[idx_frontier] = i;
            idx_frontier++;
        }
    }
    /* Update the dimension of the frontier */
    new_dim_frontier = idx_frontier;
}


/* Initialize the CSR representation of the graph when it is not read from file */
void PersonalizedPageRank::initialize_csr()
{
    neighbor_start_idx = (int *)malloc((V + 1) * sizeof(int));
    neighbors = (int *)malloc(E * sizeof(int));
    int curr_neighbor = 0;
    int curr_neighbor_start_idx = 1;
    neighbor_start_idx[0] = 0;

    for (int i = 0; i < V; i++)
    {
        for (int j = 0; j < E; j++)
        {
            if (y[j] == i)
            {
                neighbors[curr_neighbor] = x[j];
                curr_neighbor++;
            }
        }
        neighbor_start_idx[curr_neighbor_start_idx] = curr_neighbor;
        outdegrees[i] = neighbor_start_idx[curr_neighbor_start_idx] - neighbor_start_idx[curr_neighbor_start_idx - 1];
        curr_neighbor_start_idx++;
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
             true,                      // If true, read edges TRANSPOSED, i.e. edge (2, 3) is loaded as (3, 2). We set this true as it simplifies the PPR computation;
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
    //read_from_file();
    // allocate the mask to store the status of the nodes in the frontier (all false by default)
    flags = (bool *)calloc(V, sizeof(bool));
    // at the beginning the frontier contains just the personalization vertex
    // frontier = (int *)malloc(sizeof(int));
    err = cudaMallocManaged(&frontier, sizeof(int));

    //Removed for the read from file 
    err = cudaMallocManaged(&outdegrees, sizeof(int)*V); 
    initialize_csr();

    err = cudaMalloc(&neighbor_start_idx_d, sizeof(int) * (V + 1));
    err = cudaMalloc(&neighbors_d, sizeof(int) * E);
    err = cudaMalloc(&pi0_d, sizeof(double) * V);
    err = cudaMalloc(&residues_d, sizeof(double) * V);
    err = cudaMalloc(&flags_d, sizeof(bool) * V);
}



// Initialize data;
void PersonalizedPageRank::init()
{
    double eps = 1e-3;
    threshold = 1.0 / V;
    /* Compute rmax, the threshold needed to check if a node has to enter in the frontier */
    rmax = (eps / sqrt(E)) * sqrt(threshold / (((2.0 * eps / 3.0) + 2.0) * (log(2.0 / failure_probability))));
    random_walks_factor = rmax*((2*eps/3+2)*log(2*V*log(V)/failure_probability))/(threshold*eps*eps);
    cudaMemset(residues_d, 0.0, sizeof(double) * V);
    cudaMemset(pi0_d, 0.0, sizeof(double) * V);
    cudaMemcpy(neighbor_start_idx_d, neighbor_start_idx, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(neighbors_d, neighbors, sizeof(int) * E, cudaMemcpyHostToDevice);
    cudaMemcpy(flags_d, flags, sizeof(bool) * V, cudaMemcpyHostToDevice);
}


// Reset the state of the computation after every iteration.
// Reset the result, and transfer data to the GPU if necessary;
void PersonalizedPageRank::reset()
{
    // Reset the PageRank vector (uniform initialization, 1 / V for each vertex);
    std::fill(pr.begin(), pr.end(), 1.0 / V);
    // Generate a new personalization vertex for this iteration;
    personalization_vertex = rand() % V;
    //personalization_vertex = 5;
    if (debug)
        std::cout << "personalization vertex=" << personalization_vertex << std::endl;

    cudaMemset(residues_d, 0.0, sizeof(double) * V);
    cudaMemset(pi0_d, 0.0, sizeof(double) * V);
    cudaMemcpy(flags_d, flags, sizeof(bool) * V, cudaMemcpyHostToDevice);

    /*
    // Free old variables
    free(positive_residues);
    free(new_frontier);
    free(frontier);
    free(residues);
    */
}

void PersonalizedPageRank::execute(int iter)
{

    err = cudaMallocManaged(&rsum_d, sizeof(double));
    rsum_d[0] = 1.0;
    dim_frontier = 1;
    frontier[0] = personalization_vertex;
    flags[personalization_vertex] = true;
    modify_device_array_value<<<1, 1>>>(residues_d, personalization_vertex, 1.0);
    CHECK(cudaDeviceSynchronize());
    // Beginning of Forward Push step
    while (dim_frontier > 0)
    {
        int n_blocks = ceil(dim_frontier / 1024) + 1;
        int n_threads = ceil(dim_frontier / n_blocks) + 1;
        update_frontiers();
        CHECK(cudaDeviceSynchronize());
        update_pi0_and_r<<<n_blocks, n_threads>>>(frontier, alpha, pi0_d, residues_d, dim_frontier);
        CHECK(cudaDeviceSynchronize());
        frontier = new_frontier;
        dim_frontier = new_dim_frontier;
    }

    /* Retrieve the residues from the device */
    residues = (double *)malloc(sizeof(double) * V);
    cudaMemcpy(residues, residues_d, sizeof(double) * V, cudaMemcpyDeviceToHost);

    int count_positive_residues = 0;
    positive_residues = (int *)malloc(sizeof(int) * V);

    /*--- Beginning of Random Sampling step ---*/
    for (int i = 0; i < V; i++)
    {
        // stores just the nodes whose outdegree is positive
        if (residues[i] > 0 && outdegrees[i] > 0)
        {
            positive_residues[count_positive_residues] = i;
            count_positive_residues++;
        }
    }
    /* Allocate on the GPU the starting nodes of the random walks */
    err = cudaMallocManaged(&positive_residues_d, sizeof(int)*count_positive_residues);
    for(int i = 0; i < count_positive_residues; i++) {
        positive_residues_d[i] = positive_residues[i];
    }

    //free(positive_residues);

    /* --- Parallelizing for each node, random walks done sequentially --- 
    int n_blocks = ceil(count_positive_residues / 1024) + 1;
    int n_threads = ceil(count_positive_residues / n_blocks) + 1;
    random_walks<<<n_blocks, n_threads>>>(random_walks_factor, w, pi0_d, residues_d, count_positive_residues, positive_residues_d, outdegrees, neighbor_start_idx_d, alpha);
    CHECK(cudaDeviceSynchronize()); */

    /* --- Parallelizing random walks per node --- */
    generate_random_walks(count_positive_residues);
    pi0 = (double *)malloc(sizeof(double) * V);
    cudaMemcpy(pi0, pi0_d, sizeof(double) * V, cudaMemcpyDeviceToHost);

    /* Stores the PPR computed on the GPU */
    for (int i = 0; i < V; i++)
    {
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
    std::set<int> top_pr_indices;
    std::set<int> top_pr_golden_indices;
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




// Reader functions
void PersonalizedPageRank::read_from_file() {
    int read_idx = 0;
    int read_out = 0;
    int count = 0;

    FILE *f;
    char line[256];
    f = fopen("/content/drive/MyDrive/hpgda-spring22-main/cuda_CONTEST/data/demo.txt", "r");

    fgets(line, 256, f);
    fgets(line, 256, f);
    // set the number of edges
    E = atoi(line);
    neighbors = (int *)malloc(E*sizeof(int));
    printf("Neighbours number: %d\n", E);
    while (fgets(line, 256, f) != NULL)
    {
        if (line[0] != '#' && read_idx == 0 && read_out == 0)
        { // read neighbords
            neighbors[count] = atoi(line);
            count++;
        }
        else if (line[0] == '#' && read_idx == 0 && read_out == 0)
        { // end the reading of the neighbors and I will start with the idx
            fgets(line, 256, f);
            neighbor_start_idx = (int *)malloc(atoi(line) * sizeof(int));
            printf("Index number: %d\n", atoi(line));
            read_idx = 1;
            count = 0;
        }
        else if (line[0] != '#' && read_idx == 1 && read_out == 0)
        { // read index
            neighbor_start_idx[count] = atoi(line);
            count++;
        }
        else if (line[0] == '#' && read_idx == 1 && read_out == 0)
        { // end the reading of the indexes and I will start with the out
            fgets(line, 256, f);
            // set the number of vertex
            V = atoi(line);
            err = cudaMallocManaged(&outdegrees, sizeof(int)*V);
            read_out = 1;
            count = 0;
        }
        else
        { // read out
            outdegrees[count] = atoi(line);
            count++;
        }
    }
}
