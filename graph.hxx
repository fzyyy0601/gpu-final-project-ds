
#include <cassert>
#include <tuple>

#include <coo.hxx>

template<class... graph_view_t> class graph {

    graph_view_t graph_t;

    __host__ __device__ __forceinline__
    get_number_of_vertices() const {
        return graph_t.get_number_of_vertices();
    }

    __host__ __device__ __forceinline__
    get_number_of_edges() const {
        return graph_t.get_number_of_edges();
    }
};