import shutil
import pandas as pd

query_results = "D:\IIT Hyderabad\Research\Projects\KGTorrent\existing dataset\SQL_query_results\competition_notebooks_with_atleast_1_medal_and_10_votes.csv"
destination_location = "D:\IIT Hyderabad\Research\Projects\KGTorrent\existing dataset\Filltered_notebooks\competition_notebooks_with_atleast_1_medal_and_10_votes"
dataset_folder = "D:\IIT Hyderabad\Research\Projects\KGTorrent\existing dataset\KT_dataset"

query_results = pd.read_csv(query_results);

for ind in query_results.index:
     print(str(ind) + ". " + query_results['Notebook Name'][ind])
     shutil.copy(dataset_folder + "\\" + query_results['Notebook Name'][ind], destination_location)



