# gpu-final-project

# Project #8: Data Structure Library for GPUs: Graphs

## 1. Introduction
This project creates a data structure using CUDA with C++ to store a graph. The graph can 
be directed and is able to record weight for each edge. We provide two version of the data 
stucture, a CPU version (host) and a GPU version (device). Both versions use coordinate 
list (COO) method to store the adjacent matrix of the graph, which can save a lot of space 
when the graph is sparce. The CPU version stores all the information of the graph on the 
host and do all the operations sequentially on the host, while the GPU version stores the 
coordinate list on the device and accerlate operations through parallelization. Through 
testing, we find that, when the graph is big, the GPU version runs much faster for 
operations that have to traverse the whole coordinate list.
Link to project GitHub page: https://github.com/fzyyy0601/gpu-final-project-ds

## 2. Files Description
 - README.md - give an introduction to this repository
 - graph.hxx - head file storing the definition of the class graph, 
 - coo_d.hxx - head file storing the definition of the class coo_d, the GPU version of the data structure
 - coo_h.h - head file storing the definition of the class coo_h, the CPU version of the data structure
 - test.cu - cuda code to test whether the GPU version runs correctly on a very small graph
 - compare.cu - cuda code to check whether the GPU version and the CPU give the same result for same operations
 - exp.cu - cuda code to run the data structure and measure the time
 - some_is_sleeping_on_cuda2.jpg - someone has occupied the memory of the GPU on server cuda2 and not used 
   the computation resources for more than 4 days. As a result, we can not use the cuda2 to test our data structure
 - person_sleeping_on_cuda2.png - the name of the process that occupies cuda2 and does nothing
 

## 3. To Use Class graph
More examples could be found in test.cu, compare.cu, and exp.cu


### Create and Initialize a "graph" object

To create a graph of any type on host or device
    
    graph<weight_t, graph_view_t> somegraph;

    // weight_t: type of edge weight
    // graph_view_t: "coo_d" to store graph on GPU, "coo_h" to store graph on CPU

For expample, you can create a graph whose edge weight is of type "int" on GPU by

    graph<int, coo_d> g; 


To initialize a graph  

    void init(size_t* v_list,         // vertex list
              size_t v_num,           // number of vertex list
              size_t* row_idx,        // row index (source of each edge)
              size_t* col_idx,        // col index (target of each edge)
              weight_t* value,        // weight for each edge
              size_t e_num,           // number of edges
              size_t number_of_blocks,  // girdsize for kernel launch
              size_t threads_per_block, // blocksize for kernel launch
              size_t MAX);             // max memory limit

If you put the graph in the host, the number_of_blocks and threads_per_block can be any 
number and not influence the result.

For example, you can initialize a graph like this 

    g.init( {1,2,3,4,5,6,8},    // vertex list     
            7,                  // 7 vertices in total
            {1,1,4,3,5},        // sources of each edge
            {2,3,5,4,6},        // target of each edge
            {5,7,2,4,6},        // weight of each edge
            5,                  // 5 edges in total
            50,                 // 50 blocks
            256,                // 256 threads per block
            10);                // 10 vertices at most


### Change the graph
To insert a new vertex
    
    bool insert_vertex(size_t vertex);

    // if vertex is not in the graph, insert it and return true
    // otherwise, vertex is already in the graph, return false

To insert a new edge

    bool insert_edge(size_t row,
                    size_t col,
                    weight_t value);

    // if vertex row and col are already in the graph and edge row->col is not in the graph, 
       insert it and return true
    // otherwise, return false

To delete a vertex

    bool delete_vertex(size_t x);

    // if the vertex is in the graph, delete it and all related edges, and return true
    // otherwise, return false

To delete an edge

    bool delete_edge(size_t row_h,
                    size_t col_h);

    // if edge is already in the graph, delete it and return true
    // otherwise, return false


### Change the Configuration of Kernel Launch
    void modify_config(size_t number_of_blocks,
                        size_t threads_per_block);

If the graph is on host, calling this method will not influence anything.


### Get Information of the Graph
To print the graph 

    void print();

To list the information, like number of edges, of graph 

    void print_config();

To get the number of vertices in the graph

    size_t get_number_of_vertices();

To get the number of edges in the graph

    size_t get_number_of_edges();

To check whether a vertex is in the graph

    bool check_vertex(size_t vertex) ;

    // return true if vertex a is in the graph

To check whether an edge is in the graph

    bool check_edge(size_t row, size_t col);

    // return true if row -> col is in the graph

To get the weight of an edge

    weight_t get_weight(size_t row, size_t col, weight_t not_found);

    // if row -> col is in the graph, return its weight
    // otherwise, return not_found

To get number of neighbors of a vertex

    size_t get_num_neighbors(size_t x);

    // return 0 if vertex is not in the graph 

To get in-degree of a vertex

    size_t get_in_degree(size_t vertex);

    // return 0 if vertex is not in the graph 
    
To get out-degree of a vertex

    size_t get_out_degree(size_t vertex);

    // return 0 if vertex is not in the graph 

To get the list of in-neightbors of a vertex 

    std::vector<size_t> get_source_vertex(size_t vertex);

    // return empty vector if vertex is not in the graph

To get the list of out-neighbors of a vertex 

    std::vector<size_t> get_destination_vertex(size_t vertex);

    // return empty vector if vertex is not in the graph


## 4. Evironment Setting

###Demo code (exp.cu)

Compile and run the code

    nvcc -o exp exp.cu && ./exp // run the cuda code

The output should be shown as below which is the configuration you should input respectively

    GPU(0)\CPU(1) | MAX_V | v_num | e_num | test_times | MAX | grid_size | block_size

- GPU(0)\CPU(1) - To run it on GPU should input 0, on CPU should input 1
- MAX_V - The maximum vertex index in the graph (no vertex can larger than MAX_V)
- v_num - The number of vertices for initial graph
- e_num - The number of edges for initial graph
- test_times - The number of testing each funcitons
- MAX - The memory limit for graph
- grid_size - grid size when choosing GPU
- block_size - block size when choosing GPU

For example if you want to run a GPU graph on a initialized 10000 vertices 1000000 edges 
and want test the functions 1000 times, the maximum number of vertices is 20000 and memory 
limit is 2000000 with grid size 80 and block size 256. You should input

    0 20000 10000 1000000 1000 2000000 80 256

The sample result should be like

    ----------------------------Test begin------------------------------------
    Graph initialzed in GPU, grid size: 80, block size: 256
    initialization: 0.40 s
    insert 1000 vertices: 0.01 s
    insert 1000 edges: 0.03 s
    check 1000 edges: 0.02 s
    check 1000 vertices: 0.01 s
    get weight of 1000 edges: 0.03 s
    get in degree of 1000 vertices: 0.04 s
    get out degree of 1000 vertices: 0.05 s
    get num of neighbors of 1000 vertices: 0.10 s
    get source of 1000 vertices: 0.09 s
    get destination of 1000 vertices: 0.10 s
    delete 1000 edges: 0.02 s
    delete 1000 vertices: 0.04 s
    Total time: 0.94 s
    ----------------------------Test end---------------------------------------

Our code could be compiled with cuda-11.4 and gcc-4.8.5 and run correctly on the NYU cims 
server - cuda1, cuda3, cuda4, and cuda5. HOWEVER, could not test our code on cuda2 because 
someone have occupied the memory of the GPU on cuda2 and not used the computation resources 
for more than 4 DAYS!


## 5. Team Member
 - Shidong Zhang
 - Baijia Ye
 - Ziyue Feng
 - Changyue Su
