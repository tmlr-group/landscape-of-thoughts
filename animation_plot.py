from lot.animation import animate_landscape

for dataset_name in ['aqua', 'mmlu', 'strategyqa', 'commonsenseqa']:
    for model_name in ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'Meta-Llama-3.1-70B-Instruct-Turbo', 'Meta-Llama-3.1-8B-Instruct-Turbo']:
        animate_landscape(
            model_name = model_name,
            dataset_name = dataset_name,
            plot_type = 'method',
            save_root = "Landscape-Data",
            output_dir = "figures/animate_landscape",
            window_size = 200,
            step_size = 50,
            display = False
        )