import sys

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import winsound
import random


# Setting the personalization vector
def set_personalization(already_used, without_spammers_flag):
    # setting the reset vector to zeros
    personalization = dict.fromkeys(range(hostID_num), 0)
    # if needed graph without spammers this part delete the spammers slot in reset vector dictionary
    if without_spammers_flag:
        for spammer in spammers:
            del personalization[spammer]

    # this part randomly select trusted site which wasn't selected yet
    while True:
        selected_trusted_site = random.sample(trusted, 1)[0]
        if selected_trusted_site not in already_used:
            break
    # this part sets the selected site reset probability to 1
    personalization.update({selected_trusted_site: 1})
    # this part is meant to prevent the situation in which the same trusted site was selected more then once
    already_used.append(selected_trusted_site)
    return personalization


# this function creates the web graph
def create_graph(filename):
    G = nx.DiGraph()
    f = open(filename, "r")
    line = f.readline()
    hostID_num = int(line)
    for node in range(0, hostID_num):
        if node in trusted:
            color_map.append('green')
        elif node in spammers:
            color_map.append('red')
        else:
            color_map.append('blue')

        G.add_node(node)
        # if node in spammers and without_spammers_flag == 1:
        #     G.add_node(node)
        # elif node in trusted and num_of_trusted_sites > 0:
        #     G.add_node(node)
        #     # personalization.update({node: weight_of_trusted_site})
        #     # nstart.update({node: 1})
        #     # num_of_trusted_sites -= 1
        # else:
        #     G.add_node(node)
        #     # personalization.update({node: 1})
        #     # nstart.update({node: 0})

    for node in range(0, hostID_num):
        line = f.readline()
        if line != "\n":
            dests = line.split(" ")
            for dest in dests:
                dest_parts = dest.split(":")
                G.add_edge(node, int(dest_parts[0]), weight=int(dest_parts[1]))
                # if for multiDiGraph - takes a lot of time
                # for link in range(0, int(dest_parts[1])):
                #     G.add_edge(node, int(dest_parts[0]))
    return G


# this function classify the sites to trusted and spam sites
def get_assessments(filename):
    f = open(filename, "r")
    line = f.readline()
    while line != '':
        line_parts = line.split(" ")
        if line_parts[1] == "spam":
            spammers.add(int(line_parts[0]))
        elif line_parts[1] == "nonspam":
            trusted.add(int(line_parts[0]))
        line = f.readline()


# page rank function
def pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling=None):
    """Return the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
    A NetworkX graph. Undirected graphs will be converted to a directed
    graph with two directed edges for each undirected edge.

    alpha : float, optional
    Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
    The "personalization vector" consisting of a dictionary with a
    key for every graph node and nonzero personalization value for each node.
    By default, a uniform distribution is used.

    max_iter : integer, optional
    Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
    Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
    Starting value of PageRank iteration for each node.

    weight : key, optional
    Edge data key to use as weight. If None weights are set to 1.

    dangling: dict, optional
    The outedges to be assigned to any "dangling" nodes, i.e., nodes without
    any outedges. The dict key is the node the outedge points to and the dict
    value is the weight of that outedge. By default, dangling nodes are given
    outedges according to the personalization vector (uniform if not
    specified). This must be selected to result in an irreducible transition
    matrix (see notes under google_matrix). It may be common to have the
    dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
    Dictionary of nodes with PageRank as value

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence. The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.


    """
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:

        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:

        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:

            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)


# this function calculates the L1 distance between PPR and PR without spam sites
def calculate_dist_l1(no_spammers, temp_pr):
    sum_of_ranks = 0
    for i in temp_pr:
        sum_of_ranks += abs(temp_pr[i] - no_spammers[i])
    dist = sum_of_ranks
    return dist


# this function calculates the L2 distance between PPR and PR without spam sites
def calculate_dist_l2(no_spammers, temp_pr):
    sum_of_ranks = 0
    for i in range(0, len(spammers)):
        sum_of_ranks += abs(math.pow(temp_pr[i], 2) - math.pow(no_spammers[i], 2))
    dist = math.sqrt(sum_of_ranks)
    return dist


# this function calculates the norm L1 of Ranks
def calculate_norm_l1(ranks):
    norm = sum(ranks.values())
    return norm


# clear from spammers
def clear_spamers(pr):
    for spamer in spammers:
        del pr[spamer]


# delete spammers from graph
def delete_spammers_from_graph(G):
    for spamer in spammers:
        G.remove_node(spamer)


# getting spam ranks
def get_spam_ranks(pr):
    spam_ranks = dict()
    for spamer in spammers:
        spam_ranks.update({spamer: pr.get(spamer)})

    return spam_ranks


# this function plots the example of network
def print_network(test_graph_file):
    G = create_graph(test_graph_file)
    nx.draw(G, node_color=color_map, with_labels=True, font_weight='bold')
    plt.savefig("pictures/sites connections.png")
    plt.close()


# progress printer
def print_progress(iteration_num, total_num):
    progress_percentage = math.floor(100 * (iteration_num / total_num))
    text_to_print = "In progress..." + str(progress_percentage) + "%"
    sys.stdout.write('\r' + text_to_print)


# performing researched PPR
def researched_PPR(temp_G, k, without_spammers_flag):
    min_pr = dict()
    already_used = []
    if k == 0:
        min_pr = pagerank(temp_G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6,
                          nstart=None, weight='weight', dangling=None)
    else:
        for i in range(0, k):
            personalization = set_personalization(already_used, without_spammers_flag)
            temp_pr = pagerank(temp_G, alpha=0.85, personalization=personalization, max_iter=100, tol=1.0e-6,
                               nstart=None, weight='weight', dangling=None)
            if not min_pr:
                min_pr = temp_pr
            else:
                for page in temp_pr:
                    if temp_pr.get(page) < min_pr.get(page):
                        min_pr.update({page: temp_pr.get(page)})

        # normalizing the page rank
        sum_pr = sum(min_pr.values())
        for page in min_pr:
            min_pr.update({page: (min_pr.get(page) / sum_pr)})

    return min_pr


# challenge 1 - Distance L1
def challenge_1():
    ppr_diff = []
    pr_diff = []
    iteration_num = 0.0

    # this part creating the distance l1 of regular PR
    pr_without_spammers = pagerank(G_without_spammers, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6,
                                   nstart=None, weight='weight', dangling=None)
    pr_with_spammers = pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6,
                                nstart=None, weight='weight', dangling=None)
    # clearing the spammers from rank
    clear_spamers(pr_with_spammers)
    pr_dist_l1 = calculate_dist_l1(pr_without_spammers, pr_with_spammers)

    for num in num_trusted_vector:
        # printing the progress of the loop
        print_progress(iteration_num, len(num_trusted_vector))
        iteration_num += 1.0

        # performing researched PPR on graph without spammers
        ppr_without_spammers = researched_PPR(G_without_spammers, num, 1)

        # performing researched PPR on graph with spammers
        ppr_with_spammers = researched_PPR(G, num, 0)

        # clearing the spammers from rank
        clear_spamers(ppr_with_spammers)
        # calculating the distance L1 between PPR without Spammers and PPR with Spammers
        ppr_diff.append(calculate_dist_l1(ppr_without_spammers, ppr_with_spammers))
        pr_diff.append(pr_dist_l1)

    plt.figure(figsize=(8.7, 5.9))
    plt.plot(num_trusted_vector, ppr_diff, 'b', label="PPR")
    plt.plot(num_trusted_vector, pr_diff, 'r', label="PR")
    plt.legend(loc="upper right")
    plt.xlabel("Number Of Trusted Sites")
    plt.ylabel("Distance L1")
    plt.title("Distance L1 between PPR with spam and PPR without spam Vs num of trusted sites")
    plt.savefig("pictures/Distance L1 between PPR with spam and PPR without spam Vs num of trusted sites.png")
    # plt.show()
    plt.close()


# challenge 2 - Spam Resistance
def challenge_2():
    spammers_ppr = []
    spammers_pr = []
    iteration_num = 0.0

    # this part creating the spammers sum of regular PR
    pr = pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6,
                  nstart=None, weight='weight', dangling=None)

    sum_spammers_pr = calculate_norm_l1(get_spam_ranks(pr))

    for num in num_trusted_vector:
        # printing the progress of the loop
        print_progress(iteration_num, len(num_trusted_vector))
        iteration_num += 1.0

        ppr = researched_PPR(G, num, 0)

        spammers_ppr.append(calculate_norm_l1(get_spam_ranks(ppr)))
        spammers_pr.append(sum_spammers_pr)

    plt.figure(figsize=(7.7, 5.3))
    plt.plot(num_trusted_vector, spammers_ppr, 'b', label="PPR")
    plt.plot(num_trusted_vector, spammers_pr, 'r', label="PR")
    plt.legend(loc="upper right")
    plt.xlabel("Number Of Trusted Sites")
    plt.ylabel("Sum of Spam sites rank")
    plt.title("Sum of Spam sites rank Vs Trusted sites number")
    plt.savefig("pictures/Sum of Spam sites rank Vs Trusted sites number.png")
    # plt.show()
    text_tp_print = "The spam sites rank declined in " + str(abs(100 * ((spammers_ppr[len(num_trusted_vector) - 1] -
                                                                         spammers_pr[0]) / spammers_pr[0]))) + "%"
    return text_tp_print


# challenge 3 - Distortion
def challenge_3():
    stdev_pr_diff = []
    # personalization.clear()
    # nstart.clear()
    # G = create_graph(graph_file, 0, 1)
    #
    # # deleting the spam sites from the web graph
    # delete_spammers_from_graph(G)

    pr_no_trusted = pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6,
                             nstart=None, weight='weight', dangling=None)
    pr_no_trusted_stdev = statistics.stdev(pr_no_trusted.values())

    iteration_num = 0.0
    for num in num_trusted_vector:
        # printing the progress of the loop
        print_progress(iteration_num, len(num_trusted_vector))
        iteration_num += 1.0

        # # deleting the spam sites from the web graph
        # delete_spammers_from_graph(G)

        pr = researched_PPR(G, num, 0)

        # if num == 0:
        #     pr = pr_no_trusted
        # else:
        #     pr = pagerank(G, alpha=0.85, personalization=personalization, max_iter=100, tol=1.0e-6,
        #                   nstart=nstart, weight='weight', dangling=None)

        # calculating the difference between Standard Deviation of PR and PPR as an index for distortion
        stdev_pr_diff.append(statistics.stdev(pr.values()) - pr_no_trusted_stdev)

    plt.figure(figsize=(7.7, 5.3))
    plt.plot(num_trusted_vector, stdev_pr_diff, label="PPR")
    plt.legend(loc="upper right")
    plt.xlabel("Number Of Trusted Sites")
    plt.ylabel("Distortion")
    plt.title("Distortion Vs Trusted sites number")
    plt.savefig("pictures/Distortion Vs Trusted sites number.png")
    # plt.show()
    decline_percentage = abs(
        100 * ((stdev_pr_diff[len(num_trusted_vector) - 1] - max(stdev_pr_diff)) / max(stdev_pr_diff)))
    text_tp_print = "The Distortion decline from pick to max number of trusted sites is " + str(
        decline_percentage) + "%"
    # print("The Distortion decline from pick to max number of trusted sites is " + str(decline_percentage) + "%")
    return text_tp_print


def gather_network_details(filename):
    f = open(filename, "r")
    line = f.readline()
    hostID_num = int(line)
    for node in range(0, hostID_num):
        num_links = 0
        line = f.readline()
        if line != "\n":
            dests = line.split(" ")
            for dest in dests:
                dest_parts = dest.split(":")
                num_links += int(dest_parts[1])
        degree.append(num_links)

    total_number_of_links = sum(degree)
    print ("Total number of links in the web = " + str(total_number_of_links))
    max_number_of_links = max(degree)
    print ("Max number of links = " + str(max_number_of_links))
    min_number_of_links = min(degree)
    print ("Min number of links = " + str(min_number_of_links))
    avg_number_of_links = statistics.mean(degree)
    print ("Average number of links = " + str(avg_number_of_links) + "\n")
    return hostID_num


# MAIN
# initializing files names
graph_file = "hostgraph_weighted.txt"
assessments_file_1 = "assessments_1.txt"
assessments_file_2 = "assessments_2.txt"

# initializing variables
trusted = set()  # set of trusted sites
spammers = set()  # set of spam sites
color_map = []  # for graph plot
# personalization = dict()  # for PPR
nstart = dict()  # for controlling reset vector
weight_of_trusted_site = 200
degree = []

import networkx.algorithms.isomorphism as iso

# G1 = nx.DiGraph()
# G2 = nx.DiGraph()
# G1.add_path([1,2,3,4],weight=1)
# G2.add_path([1,2,3,4],weight=1)
# G2.remove_node(4)
# em = iso.numerical_edge_match('weight', 1)
# a = nx.is_isomorphic(G1, G2)  # no weights considered
# b = nx.is_isomorphic(G1, G2, edge_match=em)  # match weights
# # a = G_without_spammers == G


# plotting an example for web graph
print("Plotting an example for Webgraph...")
get_assessments("test/test_assessment.txt")
print_network("test/test_graph.txt")
trusted.clear()
spammers.clear()
print("Done")

# dealing with the real web graph
get_assessments(assessments_file_1)  # get classification for first part of sites
get_assessments(assessments_file_2)  # get classification for second part of sites
num_trusted_vector = list(np.arange(0, 18, 3))

# Gathering network details
print("Gathering network details...")
hostID_num = gather_network_details(graph_file)
print("\rDone")

# creating graph
print("Creating Graph...")
G = create_graph(graph_file)
# creating a graph without spammers
G_without_spammers = create_graph(graph_file)
# deleting the spam sites from the web graph
delete_spammers_from_graph(G_without_spammers)
print("\rDone")

# plotting the ranks of spammers as function of trusted sites number
print("Attacking the second challenge...")
spam_rank_decline = challenge_2()
print("\r" + spam_rank_decline)
print("Done")

# plotting the Distortion as function of trusted sites number
print("Attacking the third challenge...")
Distortion_Decline = challenge_3()
print("\r" + Distortion_Decline)
print("Done")

# plotting the L1 distance between PPR with spam and PPR without spam Vs number of trusted sites
# print("Attacking the first challenge...")
# challenge_1()
# print("\rDone")

# playing a beep to declare the end of the simulation
frequency = 1500  # Set Frequency To 1500 Hertz
duration = 500  # Set Duration To 500 ms == 0.5 second
winsound.Beep(frequency, duration)
