def print_as_md(overall_metrics):
    
    metrics = ["test_mr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10", "test_hits@50", "test_hits@100", "test_auc"]
    filt_metrics = [k.replace("_", "_f_") for k in metrics]

    string_metrics = "| Property | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | AUC \n"
    string_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"
    string_filtered_metrics = "| Property | MR | MRR | Hits@1 | Hits@3 | Hits@10 | Hits@50 | Hits@100 | AUC \n"
    string_filtered_metrics += "| --- | --- | --- | --- | --- | --- | --- | --- | --- | \n"
    
    string_metrics += "| Overall | "
    string_filtered_metrics += "| Overall | "
    for metric in metrics:
        if metric == "test_mr":
            string_metrics += f"{int(overall_metrics[metric])} | "
        else:
            string_metrics += f"{overall_metrics[metric]:.4f} | "
    for metric in filt_metrics:
        if metric == "test_f_mr":
            string_filtered_metrics += f"{int(overall_metrics[metric])} | "
        else:
            string_filtered_metrics += f"{overall_metrics[metric]:.4f} | "


    print(string_metrics)
    print("\n\n")
    print(string_filtered_metrics)
        
    
