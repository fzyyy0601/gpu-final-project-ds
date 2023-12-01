
#include <cassert>
#include <tuple>

#include <coo.hxx>

template<
    typename vertex_t,
    typename edge_t,
    typename weight_t,
    class... graph_view_t> class graph {

public:
    graph_view_t graph_t;

    __host__ __device__
    get_number_of_vertices() const {
        return graph_t.get_number_of_vertices();
    }

    __host__ __device__
    get_number_of_edges() const {
        return graph_t.get_number_of_edges();
    }
};