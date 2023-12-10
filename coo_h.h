#include <utility>
#include <vector>
#include <stdio.h>
#include <string.h>

template<typename weight_t> class coo_h{
    /* whether nodes i exits, v_d[i] == 1 means node i is in the graph*/
    std::vector<bool> v;
    /* source of each edge */
    std::vector<size_t> row_idx;
    /* target of each edge */
    std::vector<size_t> col_idx;
    /* wegith of each edge*/
    std::vector<weight_t> value;
    /* number of vertices */
    size_t v_num;
    /* number of edges */
    size_t e_num;
    /* max number of nodes */
    size_t MAX;
    /* deleted slots in the row_idx_d */
    std::vector<size_t> deleted;
    /* head of the deleted_d */
    size_t head;
    /* tail of the deleted_d */
    size_t tail;

public:
    /* 1 Initialize the graph */
    void init(size_t* v_list ,size_t v_num ,size_t* row_idx ,size_t* col_idx ,weight_t* value ,size_t e_num ,size_t number_of_blocks ,size_t threads_per_block ,size_t MAX){
        this->MAX=MAX;
        this->v_num=v_num;
        this->e_num=e_num;
        head=0;
        tail=0;
        v.resize(MAX);
        for(size_t i=0;i<v_num;i++){
            v[v_list[i]]=1;
        }
        this->row_idx.resize(MAX);
        memcpy(this->row_idx.data(),row_idx,e_num*sizeof(size_t));

        this->col_idx.resize(MAX);
        memcpy(this->col_idx.data(),col_idx,e_num*sizeof(size_t));

        this->value.resize(MAX);
        memcpy(this->value.data(),value,e_num*sizeof(weight_t));

        this->deleted.resize(MAX);
    }

    /* 1 print the graph */
    void print(){
        printf("---------------graph begin--------------\n");
        printf("v_num=%lu ,e_num=%lu ,head=%lu, tail=%lu ,MAX=%lu\n",v_num,e_num,head,tail,MAX);
        printf("v:  ");
        for(size_t i=0;i<MAX;i++){
            if(v[i]) printf("1 ");
            else printf("0 ");
        }
        printf("\nrow_idx: ");
        for(size_t i=0;i<e_num;i++){
            printf("%lu ",row_idx[i]);
        }
        printf("\ncol_idx: ");
        for(size_t i=0;i<e_num;i++){
            printf("%lu ",col_idx[i]);
        }
        printf("\nvalue  : ");
        for(size_t i=0;i<e_num;i++){
            printf("%d ",value[i]);
        }
        printf("\ndeleted_idx: ");
        for(size_t i=head;i<tail;i++){
            printf("%lu ",deleted[i%MAX]);
        }
        printf("\n---------------graph end----------------\n");
    }

    /* 1 modify grid size and block size*/
    void modify_config(size_t number_of_blocks,size_t threads_per_block){
    }

    /* 2 return the number of vertices, and this function is called on host */
    size_t get_number_of_vertices() {
        return v_num;
    }

    /* 2 return the number of edges, and this function is called on host */
    size_t get_number_of_edges(){
        if(head<=tail){
            return e_num-(tail-head);
        }
        else{
            return e_num-(tail-head+MAX);
        }
    }

    /* 2 if vertex is in the graph, return True. Otherwise, return False*/
    bool check_vertex(size_t vertex){
        return v[vertex];
    }

    /* 2 if edge is in the graph, return True. Otherwise, return False*/
    bool check_edge(size_t row, size_t col){
        if(!(v[row]&&v[col])){
            return false;
        }
        for(int i=0;i<e_num;i++){
            if(row_idx[i]==row&&col_idx[i]==col){
                return true;
            }
        }
        return false;
    }

    /* 2 if edge is in the graph, return the value. Otherwise, return not_found*/
    weight_t get_weight(size_t row, size_t col, weight_t not_found){
        if(!(v[row]&&v[col])){
            return not_found;
        }
        for(size_t i=0;i<e_num;i++){
            if(row_idx[i]==row&&col_idx[i]==col){
                return value[i];
            }
        }
        return not_found;
    }

    /* 4 Get neighbors */
    size_t get_num_neighbors(size_t x){
        size_t res=0;
        for(size_t i=0;i<e_num;i++){
            if(row_idx[i]==x) res++;
            if(col_idx[i]==x) res++;
        }
        return res;
    }

    /* 4 Get in degree */
    size_t get_in_degree(size_t v){
        size_t res=0;
        for(size_t i=0;i<e_num;i++){
            if(col_idx[i]==v) res++;
        }
        return res;
    }

    /* 4 Get out degree */
    size_t get_out_degree(size_t v){
        size_t res=0;
        for(size_t i=0;i<e_num;i++){
            if(row_idx[i]==v) res++;
        }
        return res;
    }

    /* 2 insert_vertex */
    bool insert_vertex(size_t vertex){
        if(v[vertex]){
            return false;
        }
        v_num += 1;
        v[vertex]=1;
        return true;
    }

    /* 1 Insert edge (row_h,col_h,value_h) into graph */
    bool insert_edge(size_t row,
                    size_t col,
                    weight_t value){
        if(!(v[row]&&v[col])){
            return false;
        }
        if(check_edge(row,col)){
            return false;
        }
        size_t idx;
        if(head==tail){
            idx=e_num;
            e_num++;
        }
        else{
            idx=head;
            head++;
            if(head==MAX){
                head=0;
            }
        }
        row_idx[idx]=row;
        col_idx[idx]=col;
        this->value[idx]=value;
        return true;
    }

    /* 3 Delete edge (row_h,col_h,value_h) */
    bool delete_edge(size_t row,
                    size_t col){
        if(!(v[row]&&v[col])){
            return false;
        }
        for(int i=0;i<e_num;i++){
            if(row_idx[i]==row&&col_idx[i]==col){
                row_idx[i]=col_idx[i]=-1;
                deleted[tail]=i;
                tail=tail+1;
                if(tail==MAX){
                    tail=0;
                }
                return true;
            }
        }
        return false;
    }

    /* 4 Delete edge (row,col,value) */
    bool delete_vertex(size_t x){
        if(!v[x]){
            return false;
        }
        for(size_t i=0;i<e_num;i++){
            if(row_idx[i]==x||col_idx[i]==x){
                row_idx[i]=col_idx[i]=-1;
                deleted[tail]=i;
                tail=tail+1;
                if(tail==MAX){
                    tail=0;
                }
                return true;
            }
        }
        return true;
    }

    /* 3 return list of destination vertex */
    std::vector<size_t> get_destination_vertex(size_t x){
        std::vector<size_t> ret;
        for(size_t i=0;i<e_num;i++){
            if(row_idx[i]==x) ret.push_back(col_idx[i]);
        }
        return ret;
    }

    /* 3 return list of source vertex */
    std::vector<size_t> get_source_vertex(size_t x){
        std::vector<size_t> ret;
        for(size_t i=0;i<e_num;i++){
            if(col_idx[i]==x) ret.push_back(col_idx[i]);
        }
        return ret;
    }
};