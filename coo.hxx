#include <utility>
#include <vector>
#include <stdio.h>
#include <cuda.h>

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
void find_edge_d(size_t row, size_t col, size_t *row_idx, size_t *col_idx, size_t e_num, bool *res){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (int i = index; i < e_num; i += stride){
        if (row_idx[i] == row && col_idx[i] == col)
            *res = 1;
    }
}

__global__ void getDegree(int *num, size_t v, size_t *idx_d, size_t n) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n && idx_d[index] == v) {
        atomicAdd((int*)*num, 1);
    }
}

/* get list of destination vertex */
__global__ void getDestinationVertex(vertex_t* list, size_t x, size_t *col_idx_d, int* count, size_t MAX_d){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < MAX_d && col_idx_d[index] == x){
        int idx = atomicAdd(count, 1);
        list[idx] = index;
    }
}

/* get list of source vertex */
__global__ void getSourceVertex(vertex_t* list, size_t x, size_t *col_idx_d, int* count, size_t MAX_d){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < MAX_d && col_idx_d[index] == x){
        int idx = atomicAdd(count, 1);
        list[idx] = index;
    }
}

template <typename weight_t>
__global__
void get_weight_d(size_t row, size_t col, size_t *row_idx, size_t *col_idx, weight_t *value, size_t e_num, weight_t *res){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (int i = index; i < e_num; i += stride){
        if (row_idx[i] == row && col_idx[i] == col)
            *res = value[i];
    }
}

template <typename weight_t>
__global__
void coo_insert_edge_d(size_t* row_idx_d,
                    size_t* col_idx_d,
                    weight_t* value_d,
                    size_t e_num_d,
                    size_t n_row_d,
                    size_t n_col_d,
                    weight_t n_value_d,
                    size_t* deleted_d,
                    size_t head_d,
                    size_t tail_d){
    int idx;
    if(head_d==tail_d){
        idx=e_num_d;
    }
    else{
        idx=deleted_d[head_d];
    }
    row_idx_d[idx]=n_row_d;
    col_idx_d[idx]=n_col_d;
    value_d[idx]=n_value_d;
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
    /* 1 Initialize the graph */
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
    }

    /* 1 print the graph */
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
        printf("---------------graph begin--------------\n");
        printf("v_num=%lu ,e_num=%lu ,head=%lu, tail=%lu ,MAX=%lu ,gridsize=%lu, blocksize=%lu\n",v_num_h,e_num_h,head_h,tail_h,MAX_h,number_of_blocks,threads_per_block);
        printf("v_d:  ");
        for(int i=0;i<MAX_h;i++){
            if(v_h[i]) printf("1 ");
            else printf("0 ");
        }
        printf("\nrow_idx_d: ");
        for(int i=0;i<e_num_h;i++){
            printf("%lu ",row_idx_h[i]);
        }
        printf("\ncol_idx_d: ");
        for(int i=0;i<e_num_h;i++){
            printf("%lu ",col_idx_h[i]);
        }
        printf("\nvalue_d  : ");
        for(int i=0;i<e_num_h;i++){
            printf("%d ",value_h[i]);
        }
        printf("\ndeleted_idx_d: ");
        for(int i=head_h;i<tail_h;i++){
            printf("%lu ",deleted_h[i%MAX_h]);
        }
        printf("\n---------------graph end----------------\n");
    }

    ~coo(){
        cudaFree(v_d);
        cudaFree(row_idx_d);
        cudaFree(col_idx_d);
        cudaFree(value_d);
        cudaFree(deleted_d);
    }

    /* 1 modify grid size and block size*/
    void modify_config(size_t number_of_blocks,size_t threads_per_block){
        this->number_of_blocks=number_of_blocks;
        this->threads_per_block=threads_per_block;
    }

    /* 2 return the number of vertices, and this function is called on host */
    size_t get_number_of_vertices() {
        return v_num_h;
    }

    /* 2 return the number of edges, and this function is called on host */
    size_t get_number_of_edges(){
        return e_num_h;
    }

    /* 2 if vertex is in the graph, return True. Otherwise, return False*/
    bool check_vertex(size_t vertex){
        bool res;
        cudaMemcpy(&res, v_d + vertex, sizeof(bool), cudaMemcpyDeviceToHost);
        return res;
    }

    /* 2 if edge is in the graph, return True. Otherwise, return False*/
    bool check_edge(size_t row, size_t col){
        if (!(check_vertex(row) && check_vertex(col))){
            return false;
        }
        bool res_h = 0;
        bool *res_d;
        cudaMalloc((void**) &res_d, sizeof(bool));
        cudaMemcpy(res_d, &res_h, sizeof(bool), cudaMemcpyHostToDevice);

        find_edge_d<<<number_of_blocks, threads_per_block>>>(row, col, row_idx_d, col_idx_d, e_num_h, res_d);
        
        cudaMemcpy(&res_h, res_d, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(&res_d);
        return res_h;
    }

    /* 2 if edge is in the graph, return the value. Otherwise, return not_found*/
    weight_t get_weight(size_t row, size_t col, weight_t not_found){
        if (!(check_vertex(row) && check_vertex(col))){
            return not_found;
        }

        weight_t res_h = not_found;
        weight_t *res_d;
        cudaMalloc((void**) &res_d, sizeof(weight_t));
        cudaMemcpy(res_d, &res_h, sizeof(weight_t), cudaMemcpyHostToDevice);

        get_weight_d<weight_t><<<number_of_blocks, threads_per_block>>>(row, col, row_idx_d, col_idx_d, value_d, e_num_h, res_d);

        cudaMemcpy(&res_h, res_d, sizeof(weight_t), cudaMemcpyDeviceToHost);
        cudaFree(&res_d);
        return res_h;
    }

    /* 1 Insert edge (row_h,col_h,value_h) into graph */
    bool insert_edge(size_t row_h,
                    size_t col_h,
                    weight_t value_h){
        if(check_edge(row_h,col_h)){
            return false;
        }
        coo_insert_edge_d<weight_t><<<1,1>>>(row_idx_d,col_idx_d,value_d,e_num_h,row_h,col_h,value_h,deleted_d,head_h,tail_h);
        if(head_h==tail_h){
            e_num_h++;
        }
        else{
            head_h++;
        }
        return true;
    }

    size_t get_out_degree(size_t v){
        size_t res = 0;
        size_t *num;
        // int *num;
        size_t vd;
        
        // memory allocation
        cudaMalloc((void **)&num, sizeof(size_t));
        //cudaMalloc((void **)&num, sizeof(int));
        cudaMalloc((void **)&vd, sizeof(size_t));
        cudaMemcpy(num, res, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(vd, v, sizeof(size_t), cudaMemcpyHostToDevice);
        getDegree<<<number_of_blocks, threads_per_block>>>(num, v, row_idx_d, MAX_h);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        //bring data back 
        cudaMemcpy(res, num, sizeof(size_t), cudaMemcpyDeviceToHost);
        // cudaMemcpy(res, num, sizeof(int), cudaMemcpyDeviceToHost);
        return res;
    }

    size_t get_in_degree(size_t v){
        size_t res = 0;
        // size_t *num;
        int* num;
        size_t vd;
        
        // memory allocation
        cudaMalloc((void **)&num, sizeof(size_t));
        // cudaMalloc((void **)&num, sizeof(int));
        cudaMalloc((void **)&vd, sizeof(size_t));
        cudaMemcpy(num, res, sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(vd, v, sizeof(size_t), cudaMemcpyHostToDevice);
        getDegree<<<number_of_blocks, threads_per_block>>>(num, v, col_idx_d, MAX_h);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(err));
        //bring data back 
        cudaMemcpy(res, num, sizeof(size_t), cudaMemcpyDeviceToHost);
        //cudaMemcpy(res, num, sizeof(int), cudaMemcpyDeviceToHost);
        return res;
    }

    /* return list of vertex */
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
        cudaFree(&cudaList);
        free(&list);
        return ret;
    }
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
        cudaFree(&cudaList);
        free(&list);
        return ret;
    }
    
};