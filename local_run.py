from experiments import *





if __name__ == '__main__':
    # freeze_support()

    """
    for graph_type in ["tree", "GNP", "smallCommunities"]:
        for minimize in [False, True]:
            plot_algo({
                "graph_type": [graph_type],
                "n": [10000],
                "num_stooges": np.linspace(5, 50, 10, dtype=int),
                "minimize": [False],
                "eps": [1e-5],
                "phi": [1.0, 1.1, 1.5, 100.0][::-1],
                "seed": range(5),
            }, "phi", ylim=(None, 260)) # , ylim=(None, 3.05))

            plot_algo({
                "graph_type": [graph_type],
                "n": [10000],
                "num_stooges": np.linspace(5, 50, 10, dtype=int),
                "minimize": [False],
                "eps": [1e-8, 1e-5, 1e-2, 1.0],
                "phi": [1.1],
                "seed": range(5),
            }, "eps", ylim=(None, 260)) # , ylim=(0.015, 1.85))
    """

    """
    plot_brute_force({
        "n": [10],
        "num_stooges": [5],
        "seed": range(5, 10),
    })

    plot_brute_force({
        "n": [15],
        "num_stooges": [5],
        "seed": range(10, 15),
    })

    plot_brute_force({
        "n": [20],
        "num_stooges": [10],
        "seed": range(5, 10),
    })
    """

    """
    plot_scalability({
        "n": np.linspace(100, 500, 10).astype(int),
        "num_stooges": [50],
        "minimize": [True],
        "method": ["random", "maxdeg", "centrality", "greedy", "naive-greedy"],
        "seed": range(5),
    })
    """

    """
    dists_plot({
        "graph_type": ["grid"],
        "init_type": [None],
        "num_stooges": [10],
        "minimize": [True],
        "method": ["random", "maxdeg", "centrality", "greedy"],
        "polarization": [False],
        "seed": range(1),
    })

    dists_plot({
        "graph_type": ["grid"],
        "init_type": [None],
        "num_stooges": [10],
        "minimize": [False],
        "method": ["random", "maxdeg", "centrality", "greedy"],
        "polarization": [False],
        "seed": range(1),
    })
    """


    synthetic_graph_types = ["GNP", "tree", "smallCommunities", "grid", "d_regular", "star_random", "STAR"]
    synthetic_graph_labels = ["GNP$(150, 0.05)$", "RndTree(150)", "RndCommunities", "Grid", "d_regular", "star_random", "STAR"]
    synthetic_graph_types2 = synthetic_graph_types[:4]

    """
    test_isect_pol_mse({
        "graph_type": synthetic_graph_types2,
        "minimize": [True],
        "seed": range(5),
    })

    test_isect_pol_mse({
        "graph_type": synthetic_graph_types2,
        "minimize": [False],
        "seed": range(5),
    })
    """


    """
    test_isect_pol_mse_change({
        "graph_type": synthetic_graph_types2,
        "minimize": [True],
        "num_stooges": np.linspace(5, 50, 10, dtype=int),
        "seed": range(5),
    }, has_legend=False)

    test_isect_pol_mse_change({
        "graph_type": synthetic_graph_types2,
        "minimize": [False],
        "num_stooges": np.linspace(5, 50, 10, dtype=int),
        "seed": range(5),
    })
    """


    """
    plot_synthetic({
        "graph_type": synthetic_graph_types,
        "init_type": [None, "uniform", "gaussian", "exponential"],
        "num_stooges": [50],
        "minimize": [True, False],
        "method": ["random", "maxdeg", "centrality", "greedy"],
        "polarization": [False],
        "seed": range(5),
    })
    """

    """
    plot_synthetic({
        "graph_type": ["smallCommunities"],
        "init_type": ["uniform", "gaussian", "exponential"],
        "num_stooges": [50],
        "minimize": [False],
        "method": ["random", "maxdeg", "centrality", "greedy"],
        "polarization": [False],
        "seed": range(5),
    }, side_by_side=True)
    """


    """
    for graph_label, graph_type in zip(synthetic_graph_labels, synthetic_graph_types):
        for minimize in [True, False]:
            for polarization in [True, False]:
                plot_synthetic({
                    "graph_type": [graph_type],
                    "init_type": [None],
                    "num_stooges": [50],
                    "minimize": [minimize],
                    "method": ["random", "maxdeg", "centrality", "greedy"],
                    "polarization": [polarization],
                    "seed": range(5),
                }, has_legend=not minimize, title=graph_label)

            plot_synthetic_opinions({
                "graph_type": [graph_type],
                "init_type": [None],
                "num_stooges": [20],
                "minimize": [minimize],
                "method": ["greedy"],
                "polarization": [True, False],
                "seed": range(1),
            }, title=graph_label)

            for init_type in ["uniform", "gaussian", "exponential"]:
                plot_synthetic({
                    "graph_type": [graph_type],
                    "init_type": [init_type],
                    "num_stooges": [50],
                    "minimize": [minimize],
                    "method": ["random", "maxdeg", "centrality", "greedy"],
                    "polarization": [False],
                    "seed": range(5),
                })
    """



    real_world_datasets = ["war", "vax", "vaxnovax-retweet", "leadersdebate-follow", "leadersdebate-retweet", "russia_march-follow", "russia_march-retweet", "baltimore-follow", "baltimore-retweet", "beefban-follow", "beefban-retweet", "gunsense-follow", "gunsense-retweet", "vaxnovax-follow"]


    plot_real_world_change({
        "dataset": real_world_datasets,
        "minimize": [True, False],
        "method": ["random", "maxdeg", "centrality", "greedy"],
        "polarization": [True, False],
        "seed": range(6),
    })

    """
    for dataset in real_world_datasets:
        for minimize in [True, False]:

            plot_real_world_opinions({
                "dataset": [dataset],
                "minimize": [minimize],
                "method": ["greedy"],
                "polarization": [True, False],
                "seed": range(1),
            }, show_diff=False)

            for polarization in [True, False]:
                plot_real_world_change({
                    "dataset": [dataset],
                    "minimize": [minimize],
                    "method": ["random", "maxdeg", "centrality", "greedy"],
                    "polarization": [polarization],
                    "seed": range(5),
                }, has_legend=not minimize)

                plot_real_world_change({
                    "dataset": [dataset],
                    "minimize": [minimize],
                    "method": ["greedy"],
                    "polarization": [polarization],
                    "seed": range(5),
                }, has_legend=not minimize, show_decomp=True)
    """


