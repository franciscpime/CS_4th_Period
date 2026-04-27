import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    # Ex: the probability if we know nothing about that person’s parents
    #   >>> there’s a 1% chance of having 2 copies of the gene
    #   >>> 3% chance of having 1 copy of the gene
    #   >>> 96% chance of having 0 copies of the gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        # So PROBS["trait"][2] is the probability distribution that a person has 
        # the trait given that they have two versions of the gene: 
        #   >>> 65% chance of exhibiting the trait
        #   >>> 35% chance of not exhibiting the trait
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        # If a person has 0 copies of the gene, they have:
        #   >>> 1% chance of exhibiting the trait
        #   >>> 99% chance of not exhibiting the trait
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    # If a mother has two versions of the gene and passes one on to her child:
    #   >>> 1% chance it mutates into not being the target gene anymore. 
    # 
    # If a mother has no versions of the gene, and doesn't pass it:
    #   >>> 1% chance it mutates into being the target gene. 
    # 
    # It’s therefore possible that even if neither parent has any copies 
    # of the gene in question, their child might have:
    #   >>> 1 or even 2 copies of the gene
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")

    # loads data from a file into a dictionary "people"
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    # Ex:
    #   >>> probabilities["Harry"]["gene"][1] 
    #       will be the probability that Harry has 1 copy of the gene

    #   >>> probabilities["Lily"]["trait"][False] 
    #       will be the probability that Lily does not exhibit the trait
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


"""
Load gene and trait data from a file into a dictionary.
File assumed to be a CSV containing fields name, mother, father, trait.
mother, father must both be blank, or both be valid names in the CSV.
trait should be 0 or 1 if trait is known, blank otherwise.
"""
def load_data(filename):
    
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


"""
Compute and return a joint probability.

The probability returned should be the probability that
    * everyone in set `one_gene` has one copy of the gene, and
    * everyone in set `two_genes` has two copies of the gene, and
    * everyone not in `one_gene` or `two_gene` does not have the gene, and
    * everyone in set `have_trait` has the trait, and
    * everyone not in set` have_trait` does not have the trait.
"""
def joint_probability(people: dict, one_gene: set, two_genes: set, have_trait: set):
   # Joint probability
   prob = 1
   
   for person in people:
    if person in one_gene:
        num_genes = 1
    elif person in two_genes:
        num_genes = 2
    else:
        num_genes = 0

    if person in have_trait:
        has_trait = True
    else:
        has_trait = False

    mother = people[person]["mother"]
    father = people[person]["father"]

    if (people[person]["mother"] is None 
        and people[person]["father"] is None):
        prob *= PROBS["gene"]
        prob *= PROBS["trait"][num_genes][has_trait]
    else:
        if mother in one_gene:
            mum_gene = 1
        elif mother in two_genes:
            mum_gene = 2
        else:
            mum_gene = 0

        if father in one_gene:
            dad_gene = 1
        elif father in two_genes:
            dad_gene = 2
        else:
            dad_gene = 0
        
        if mum_gene == 1:
            mum_prob = 0.5
        elif mum_gene == 2:
            mum_prob = 1 - PROBS["mutation"]
        else:
            mum_prob = PROBS["mutation"]

        if dad_gene == 1:
            dad_prob = 0.5
        elif dad_gene == 2:
            dad_prob = 1 - PROBS["mutation"]
        else:
            dad_prob = PROBS["mutation"]
            
        if num_genes == 2:
            comb_prob = mum_prob * dad_prob
        elif num_genes == 1:
            comb_prob = (1-mum_prob) * dad_prob + mum_prob * (1-dad_prob)
        else:
            comb_prob = (1-mum_prob) * (1-dad_prob)

        prob *= comb_prob * PROBS["trait"][num_genes][has_trait]

    return prob



def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
