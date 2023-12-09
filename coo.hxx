#include <utility>
#include <vector>
#include <stdio.h>
#include <cuda.h>

template <typename weight_t>
__global__
void coo_init_v_d(bool *v_d,size_t *v_list_d,size_t n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int wd=gridDim.x*blockDim.x;
    while(index<n){
        v_d[v_list_d[index]]=1;
        index+=wd;
    }
}

__global__ 
void getSourceVertex(vertex_t* list, size_t x, size_t *col_idx_d, int* count, size_t MAX_d){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < MAX_d && col_idx_d[index] == x){
        int idx = atomicAdd(count, 1);
        list[idx] = index;
    }
}

__global__ 
void getDestinationVertex(vertex_t* list, size_t x, size_t *col_idx_d, int* count, size_t MAX_d){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < MAX_d && col_idx_d[index] == x){
        int idx = atomicAdd(count, 1);
        list[idx] = index;
    }
}

template <typename weight_t>
__global__
void get_weight_d(size_t row, size_t col, weight_t *res){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int strid = blockDim.x * gridDim.x;
    for (int i = index; i < e_num_d; i += strid){
        // if (row_idx_d[i] == edge[0] && col_idx_d[i] == edge[1])
        if (row_idx_d[i] == row && col_idx_d[i] == size)
            // *res = value_d[i];
            cudaMemcpy(res, value_d + i, sizeof(weight_t), cudaMemcpyDeviceToDevice);
    }
}

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
    size_t v_num_h;
    /* number of edges */
    size_t e_num_h;
    /* max number of nodes */
    size_t MAX_h;
    /* deleted slots in the row_idx_d */
    size_t *deleted_d;
    /* head of the deleted_d */
    size_t head_h;
    /* tail of the deleted_d */
    size_t tail_h;
    /* number of blocks */
    size_t number_of_blocks;
    /* number of threads per block */
    size_t threads_per_block;

public:
    /*
    1
    */
    void init(size_t* v_list_h ,size_t v_num_h ,size_t* row_idx_h ,size_t* col_idx_h ,weight_t* value_h ,size_t e_num_h ,size_t number_of_blocks ,size_t threads_per_block ,size_t MAX_h){
        this->number_of_blocks=number_of_blocks;
        this->threads_per_block=threads_per_block;
        this->MAX_h=MAX_h;
        this->v_num_h=v_num_h;
        this->e_num_h=e_num_h;
        head_h=0;
        tail_h=0;

        cudaMalloc((void **)&v_d,MAX_h*sizeof(bool));
        cudaMemset((void **)&v_d,0,MAX_h*sizeof(bool));
        size_t *v_list_d;
        cudaMalloc((void **)&v_list_d,v_num_h*sizeof(size_t));
        cudaMemcpy(v_list_d,v_list_h,v_num_h*sizeof(size_t),cudaMemcpyHostToDevice);
        coo_init_v_d<<<number_of_blocks,threads_per_block>>>(v_d,v_list_d,v_num_h);
        cudaDeviceSynchronize();
        cudaFree(v_list_d);

        cudaMalloc((void **)&row_idx_d,MAX_h*sizeof(size_t));
        cudaMemcpy(row_idx_d,row_idx_h,e_num_h*sizeof(size_t),cudaMemcpyHostToDevice);

        cudaMalloc((void **)&col_idx_d,MAX_h*sizeof(size_t));
        cudaMemcpy(col_idx_d,col_idx_h,e_num_h*sizeof(size_t),cudaMemcpyHostToDevice);

        cudaMalloc((void **)&value_d,MAX_h*sizeof(weight_t));
        cudaMemcpy(value_d,value_h,e_num_h*sizeof(weight_t),cudaMemcpyHostToDevice);

        cudaMalloc((void **)&deleted_d,MAX_h*sizeof(size_t));

        print();
    }

    void print(){
        bool v_h[MAX_h];
        size_t row_idx_h[MAX_h];
        size_t col_idx_h[MAX_h];
        weight_t value_h[MAX_h];
        size_t deleted_h[MAX_h];
        cudaMemcpy(v_h,v_d,MAX_h*sizeof(bool),cudaMemcpyDeviceToHost);
        cudaMemcpy(row_idx_h,row_idx_d,MAX_h*sizeof(size_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(col_idx_h,col_idx_d,MAX_h*sizeof(size_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(value_h,value_d,MAX_h*sizeof(weight_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(deleted_h,deleted_d,MAX_h*sizeof(size_t),cudaMemcpyDeviceToHost);
        printf("v_d: ");
        for(int i=0;i<MAX_h;i++){
            if(v_h[i]) printf("1 ");
            else printf("0 ");
        }
        printf("\nrow_idx_d: ");
        for(int i=0;i<MAX_h;i++){
            printf("%lu ",row_idx_h[i]);
        }
        printf("\ncol_idx_d: ");
        for(int i=0;i<MAX_h;i++){
            printf("%lu ",col_idx_h[i]);
        }
        printf("\nvalue_d  : ");
        for(int i=0;i<MAX_h;i++){
            printf("%d ",value_h[i]);
        }
        printf("\nrow_idx_d: ");
        for(int i=head_h;i<tail_h;i++){
            printf("%lu ",deleted_h[i%MAX_h]);
        }
    }

    ~coo(){
        cudaFree(v_d);
        cudaFree(row_idx_d);
        cudaFree(col_idx_d);
        cudaFree(value_d);
        cudaFree(deleted_d);
    }

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
    std::vector<vertex_t> get_source_vertex(size_t x){
        int num = get_in_degree(x);
        //int count = 0;
        int* d_count;
        vertex_t* list = (vertex_t*)malloc(num* sizeof(vertex_t));
        vertex_t* cudaList;
        cudaMalloc(&cudaList, sizeof(vertex_t)* num);
        //cudaMalloc(&d_count, sizeof(int));
        //cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
        getSourceVertex<<<number_of_blocks, threads_per_block>>>(cudaList, x, row_idx_d, d_count, MAX_d);
        cudaMemcpy(list, cudaList, sizeof(vertex_t)* num, cudaMemcpyDeviceToHost);
        //cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<vertex_t> ret(num);
        for(int i = 0; i < num; i++){
            ret[i] = list[i];
        }
        return ret;
    }

    /*
    3
    */
    std::vector<vertex_t> get_destination_vertex(size_t x){
        int num = get_out_degree(x);
        //int count = 0;
        int* d_count;
        vertex_t* list = (vertex_t*)malloc(num* sizeof(vertex_t));
        vertex_t* cudaList;
        cudaMalloc(&cudaList, sizeof(vertex_t)* num);
        //cudaMalloc(&d_count, sizeof(int));
        //cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
        getDestinationVertex<<<number_of_blocks, threads_per_block>>>(cudaList, x, col_idx_d, d_count, MAX_d);
        cudaMemcpy(list, cudaList, sizeof(vertex_t)* num, cudaMemcpyDeviceToHost);
        //cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<vertex_t> ret(num);
        for(int i = 0; i < num; i++){
            ret[i] = list[i];
        }
        return ret;
    }

    // /*
    // 2
    // */
    // edge_t get_weight(edge_t){
    
    /* 2 get the weight of a vertix */
    __host__
    weight_t get_weight(size_t row, size_t col){
        weight_t res_h;
        weight_t res_d;
        // size_t *edge_h[2];
        // edge_h[0] = row;
        // edge_h[1] = col;
        // size_t *edge_d;
        // cudaMalloc((void**) &edge_d, 2 * sizeof(size_t));
        cudaMalloc((void*) &res_d, sizeof(weight_t));
        // cudaMemcpy(edge_d, edge_h, 2 * sizeof(size_t), cudaMemcpyHostToDevice);
        get_weight_d<<<number_of_blocks, threads_per_block>>><weight_t>(row, col, &res_d);
        cudaMemcpy(&res_h, res_d, sizeof(weight_t), cudaMemcpyDeviceToHost);
        
        return res_h;
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
    int get_out_degree(vertex_t){
         size_t num;
        // kernel: for each atomic add, *row_idx_d;       
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