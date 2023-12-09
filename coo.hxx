#include <utility>
#include <vector>
#include <stdio.h>
#include <cuda.h>

__global__
void coo_init_v_d(bool *v_d,size_t *v_list_d,size_t n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int wd=gridDim.x*blockDim.x;
    printf("index - %d\n",index);
    while(index<n){
        printf("index\n");
        v_d[v_list_d[index]]=1;
        index+=wd;
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
};