#include <utility>
#include <vector>

template<typename weight_t> class coo{
    /* whether nodes i exits, v_d[i] == 1 means node i is in the graph*/
    bool *v_d;
    /* source of each edge */
    size_t *row_idx_d;
    /* target of each edge */
    size_t *col_idx_d;
    /* wegith of each edge*/
    weight_t *value_d;
    /* number of vertices */
    size_t v_num_d;
    /* number of edges */
    size_t e_num_d;
    /* max number of nodes */
    size_t MAX_d;
    /* deleted slots in the row_idx_d */
    size_t *deleted_d;
    /* head of the deleted_d */
    size_t head;
    /* tail of the deleted_d */
    size_t tail;
    int number_of_blocks;
    int threads_per_block;
public:
    /*s
    1
    */
    __host__ 
    void init(size_t* v_list_t ,size_t v_num_t ,size_t* row_idx_t ,size_t* col_idx_t ,size_t* value_t ,size_t e_num_t ,size_t MAX){
        cudaMalloc();
        cudaMalloc();
        cudaMalloc();
    }

    __device__
    void init_d(){}

    /* 2 return the number of vertices, and this function is called on host */
    __host__
    size_t get_number_of_vertices() {
        size_t res;
        cudaMemcpy(&res, v_num_d, sizeof(size_t), cudaMemcpyDeviceToHost);
        return res;
    }

    /* 3 return the number of edges, and this function is called on host */
    __host__
    size_t get_number_of_edges(){
        size_t res;
        cudaMemcpy(&res, e_num_d, sizeof(size_t), cudaMemcpyDeviceToHost);
        return res;
    }
//     /*
//     4
//     */
    __host__ __device__
    int get_num_neighbors(vertex_type x){
        size_t in_num = get_in_degree(x);
        size_t out_num = get_out_degree(x);
        return in_num + out_num;
    }

    // /*
    // 1
    // */
    // void set(){

    // }
    // /*
    // 2 deleted
    // */
    // vertex_t get_source_vertex(){

    // }

    /*
    3
    */
    // vertex_t get_destination_vertex(){

    // }
    // /*
    // 2
    // */
    // edge_t get_weight(edge_t){
    
    /* 2 get the weight of a vertix */
    __host__
    weight_t get_weight(size_t row, size_t col){
        weight_t res;
        size_t *edge;
        cudaMalloc((void**) &edge, 2 * sizeof(size_t));
        return res;
    }
    
    __device__
    void get_weight_d(size_t row, size_t col){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        int strid = blockDim.x * gridDim.x;


    }


    // /*
    // 3
    // */
    int get_in_degree(vertex_t v){
        size_t res = 0;
        size_t *num;
        vertex_t *vd;
        
        // memory allocation
        cudaMalloc((void **)&num, sizeof(size_t));
        cudaMalloc((void **)&vd, sizeof(vertex_t));
        cudaMemcpy(num, res, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(vd, v, sizeof(vertex_t), cudaMemcpyHostToDevice);
        getDegree<<<number_of_blocks, threads_per_block>>>(num, v, col_idx_d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        //bring data back 
        cudaMemcpy(res, num, sizeof(size_t), cudaMemcpyDeviceToHost);
        return res;
    }
    // /*
    // 4
    // */
    int get_out_degree(vertex_t v){
        size_t res = 0;
        size_t *num;
        vertex_t *vd;
        
        // memory allocation
        cudaMalloc((void **)&num, sizeof(size_t));
        cudaMalloc((void **)&vd, sizeof(vertex_t));
        cudaMemcpy(num, res, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(vd, v, sizeof(vertex_t), cudaMemcpyHostToDevice);
        getDegree<<<number_of_blocks, threads_per_block>>>(num, v, row_idx_d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        //bring data back 
        cudaMemcpy(res, num, sizeof(size_t), cudaMemcpyDeviceToHost);
        return res;
    }
    __global__ void getDegree(size_t *num, vertex_t v, size_t *idx_d) {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < n && idx_d[index] == x) {
            atomicAdd(num, 1);
        }
    }

    // /*
    // 1
    // */
    // edge_t insert_edge(edge_t){

    // }
    // /*
    // 2
    // */
    // vertex_t insert_vertex(edge_t ){

    // }
    // /*
    // 3
    // */
    // edge_t delete_edge(edge_t){

    // }
    // /*
    // 4
    // */
    vertex_t delete_vertex(vertex_t *v_del){
        for(auto v : v_del) {
            v_d[v] = 0;
        }
        // kernel 1 : v_d from 1 to 0
        // kernel 2 : *row_idx_d and *col_idx_d; find the vertex to delete 
        //            from value to -1

    }
};