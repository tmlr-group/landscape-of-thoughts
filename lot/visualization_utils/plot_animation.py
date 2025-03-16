
def plot_chain_animation(num_chains, num_thoughts_each_chain, coordinates_2d, anchors_idx_x, labels_anchors, answer_gt_short):
    # Extract the lines based on num_thoughts_each_chain
    lines = []
    for chain_idx in range(num_chains):
        start_idx = sum(num_thoughts_each_chain[:chain_idx])
        end_idx = sum(num_thoughts_each_chain[:chain_idx + 1])
        x = list(coordinates_2d[start_idx:end_idx, 0])
        y = list(coordinates_2d[start_idx:end_idx, 1])
        lines.append((x, y))

    # Calculate the global max x and y values
    max_x = max(max(x) for x, _ in lines) + 5
    max_y = max(max(y) for _, y in lines) + 5

    min_x = min(min(x) for x, _ in lines) - 5
    min_y = min(min(y) for _, y in lines) - 5

    # Determine the maximum step count
    max_steps = max(len(x) for x, _ in lines)

    # Create a base figure
    fig = go.Figure()

    # NOTE: [global maximum steps] Normalized indices to use for the color scale
    # normalized_indices = np.linspace(0, 1, max_steps)

    # Add initial traces for each line
    for i, (x, y) in enumerate(lines):
        # NOTE: [local maximum steps] Normalized indices to use for the color scale
        normalized_indices = np.linspace(0, 1, len(x))
        
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode='markers',
            name=f'Chain {i+1}',
            line=dict(width=2),
            marker=dict(
                size=5, 
                color=normalized_indices,  # Color based on index within the chain
                colorscale='RdYlGn',  # Choose a colorscale (Viridis is an example)
                showscale=True,
            ), # green or red
            showlegend=False,
        ))

    colors = px.colors.qualitative.Plotly
    for idx, anchor_name in enumerate(labels_anchors):
        if anchor_name == 'Start': # start
            marker_symbol = 'star'
        elif anchor_name == answer_gt_short: 
            # correct answer
            marker_symbol='diamond'
        else: # negative answer 
            marker_symbol = 'x'

        fig.add_trace(
            go.Scatter(
                x=[coordinates_2d[anchors_idx_x[idx], 0]], y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                mode='markers',
                name=anchor_name, marker=dict(symbol=marker_symbol, size=20, line_width=1, color=colors[idx % len(colors)])
        ))


    # Create frames
    frames = []
    for step in range(1, max_steps + 1):
        frame_data = []
        for i, (x, y) in enumerate(lines):
            # Limit the subset to the length of the current line
            if step <= len(x):
                frame_data.append(go.Scatter(
                    x=x[:step], y=y[:step],
                    mode='markers',
                    name=f'Chain {i+1}',
                    line=dict(width=2),
                    showlegend=False,
                ))
            else:
                # If current step exceeds line length, plot the entire line
                frame_data.append(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name=f'Chain {i+1}',
                    line=dict(width=2),
                    showlegend=False,
                ))
        
        # Include anchors in every frame
        colors = px.colors.qualitative.Plotly
        for idx, anchor_name in enumerate(labels_anchors):
            if anchor_name == 'Start': # start
                marker_symbol = 'star'
            elif anchor_name == answer_gt_short: 
                # correct answer
                marker_symbol='diamond'
            else: # negative answer 
                marker_symbol = 'x'

            frame_data.append(
                go.Scatter(
                    x=[coordinates_2d[anchors_idx_x[idx], 0]], y=[coordinates_2d[anchors_idx_x[idx], 1]], 
                    mode='markers',
                    name=anchor_name, marker=dict(symbol=marker_symbol, size=20, line_width=1, color=colors[idx % len(colors)])
            ))
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ), margin=dict(l=10, r=10, t=20, b=10),)

        frames.append(go.Frame(data=frame_data, name=str(step)))

    fig.frames = frames

    # Set layout with sliders, buttons, and dynamic axis ranges
    fig.update_layout(
        xaxis=dict(range=[min_x, max_x]),  # Dynamic x-axis range
        yaxis=dict(range=[min_y, max_y]),  # Dynamic y-axis range
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate', fromcurrent=True)]
                    ),
                ],
            ),
        ],
        sliders=[dict(steps=[dict(method='animate',
                                args=[[str(k)], dict(mode='immediate', frame=dict(duration=500, redraw=True), fromcurrent=True)],
                                label=str(k)) for k in range(1, max_steps + 1)],
                    transition=dict(duration=0))],
    )

    # Show the figure
    fig.show()