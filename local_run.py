from experiments import *





if __name__ == '__main__':
    # freeze_support()


    """
    plot_scalability({
        "n": np.linspace(100, 500, 10).astype(int),
        "num_stooges": [50],
        "minimize": [True],
        "method": ["random", "maxdeg", "centrality", "greedy", "naive-greedy"],
        "seed": range(5),
    })
    """

    dists_plot({
        "graph_type": ["grid"],
        "init_type": [None],
        "num_stooges": [10],
        "minimize": [True],
        "method": ["random", "maxdeg", "centrality", "greedy"],
        "seed": range(1),
    })

    dists_plot({
        "graph_type": ["grid"],
        "init_type": [None],
        "num_stooges": [10],
        "minimize": [False],
        "method": ["random", "maxdeg", "centrality", "greedy"],
        "seed": range(1),
    })

    """

    for graph_type in ["GNP", "d_regular", "star_random", "tree", "smallCommunities", "STAR", "grid"]:
        for minimize in [True, False]:
            plot_synthetic({
                "graph_type": [graph_type],
                "init_type": [None],
                "num_stooges": [50],
                "minimize": [minimize],
                "method": ["random", "maxdeg", "centrality", "greedy"],
                "seed": range(5),
            })

            plot_synthetic_opinions({
                "graph_type": [graph_type],
                "init_type": [None],
                "num_stooges": [20],
                "minimize": [minimize],
                "method": ["greedy"],
                "seed": range(1),
            })


    for dataset in ["vaxnovax-follow", "vaxnovax-retweet", "leadersdebate-follow", "leadersdebate-retweet", "russia_march-follow", "russia_march-retweet", "baltimore-follow", "baltimore-retweet", "beefban-follow", "beefban-retweet", "gunsense-follow", "gunsense-retweet", "vax"]:

        for minimize in [True, False]:
            plot_real_world_opinions({
                "dataset": [dataset],
                "minimize": [minimize],
                "method": ["greedy"],
                "seed": range(1),
            })

            plot_real_world_change({
                "dataset": [dataset],
                "minimize": [minimize],
                "method": ["random", "maxdeg", "centrality", "greedy"],
                "seed": range(5),
            })

    """


