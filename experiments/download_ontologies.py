import os
import wget

script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(script_dir, "use_cases/")

def main():
    go_url = "https://release.geneontology.org/2024-04-24/ontology/go.owl"
    foodon_url = "https://github.com/FoodOntology/foodon/releases/download/v2024-04-07/foodon.owl"
    
    os.makedirs(root_dir, exist_ok=True)

    wget.download(go_url, root_dir)
    wget.download(foodon_url, root_dir)
    
if __name__ == "__main__":
    main()
    
