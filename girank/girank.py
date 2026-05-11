def girank(family_tree, damping=0.85, iterations=100):
    # Collect all unique people in the tree
    all_people = set()
    for parent, children in family_tree.items():
        all_people.add(parent)
        for child in children:
            all_people.add(child)

    n = len(all_people)

    # Initialize every person with equal rank
    rank = {person: 1 / n for person in all_people}

    # Build a reverse map: for each person, who are their parents?
    parents_of = {person: [] for person in all_people}

    for parent, children in family_tree.items():
        for child in children:
            parents_of[child].append(parent)

    # Iterative GIRank calculation
    for _ in range(iterations):
        new_rank = {}
        for person in all_people:
            # Base rank from damping (like the random surfer in PageRank)
            inherited = (1 - damping) / n

            # Add influence from each parent
            for parent in parents_of[person]:
                num_children = len(family_tree[parent])
                # Parent splits influence equally among all children
                inherited += damping * (rank[parent] / num_children)

            new_rank[person] = inherited

        rank = new_rank

    return rank


def print_girank(rank):
    """Prints the GIRank results sorted from most to least influential."""
    sorted_rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Genetic Influence Rank (GIRank) ===\n")

    for i, (person, score) in enumerate(sorted_rank, 1):
        print(f"  {i}. {person:<10} >> {score:.6f}")

    print()

family_tree = {
    "Alice":   ["Bob", "Charlie"],
    "Bob":     ["David"],
    "Charlie": ["Eve", "Frank"],
    "David":   ["George"],
    "Eve":     ["Hannah"],
    "Frank":   ["Isaac"],
}

rank = girank(family_tree)
print_girank(rank)
