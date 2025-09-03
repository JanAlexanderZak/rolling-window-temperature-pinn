import plotly.graph_objects as go

from helpers import generate_mockup_loss
from rolling_window_scheduler import MockupRollingReduceLROnPlateauRollingWindow
from temperature_dependent_update import compare_efficiency


if __name__ == "__main__":

    # *
    # * Rolling window scheduler
    # *

    epochs = 120
    epochs_range = list(range(epochs))
    base_reduction_factor = 1.
    losses = []

    scheduler = MockupRollingReduceLROnPlateauRollingWindow(
        initial_lr=0.01,
        factor=0.9,
        patience=20,
        rolling_window=20,
        threshold=0,
    )

    for epoch in range(epochs):
        loss = generate_mockup_loss(epoch, base_reduction_factor)
        losses.append(loss)

        lr_reduced = scheduler.step(loss, epoch)
        
        if lr_reduced:
            base_reduction_factor *= scheduler.factor
            print(f"Applied loss reduction factor: {scheduler.factor}, new base factor: {base_reduction_factor:.6f}.")

    reduction_losses = [losses[epoch] for epoch in scheduler.reduction_epochs]

    fig = go.Figure()

    fig = fig.add_trace(
        go.Scatter(
            x=epochs_range, 
            y=losses,
            mode='lines',
            name='Loss',
            line=dict(color='steelblue', width=3),
            hovertemplate='<b>Loss</b><br>Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
        ),
    )

    fig = fig.add_trace(
        go.Scatter(
            x=epochs_range,
            y=scheduler.rolling_bests,
            mode='lines',
            name='Rolling Window Best (20 epochs)',
            line=dict(color='forestgreen', width=2, dash='dash'),
            hovertemplate='<b>Rolling Window Best</b><br>Epoch: %{x}<br>Best in window: %{y:.6f}<extra></extra>'
        )
    )
    
    fig = fig.add_trace(
        go.Scatter(
            x=scheduler.reduction_epochs,
            y=reduction_losses,
            mode='markers',
            name='LR reduction',
            marker=dict(
                color='red',
                size=12,
                symbol='triangle-down',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='<b>LR reduction</b><br>Epoch: %{x}<br>Loss: %{y:.6f}<br>Both LR and loss reduced by factor 0.1<extra></extra>'
        ),
    )

    fig = fig.add_shape(
        type="rect",
        x0=24, x1=43, y0=0, y1=max(losses),
        fillcolor="yellow", opacity=0.2,
        layer="below", line_width=0,
    )

    fig = fig.add_annotation(
        x=33, y=max(losses) * 0.9,
        text="Plateau",
        showarrow=False,
        font=dict(color="orange", size=12),
    )

    fig = fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis2=dict(
            title="Learning Rate",
            overlaying="y",
            side="right",
            type="log",
            showgrid=False,
        ),
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        font=dict(size=20),
        width=1000, height=600,
    )

    fig.write_image("src/plots/rolling_window_scheduler.png", scale=2)
    fig.write_html("src/plots/rolling_window_scheduler.html")
    fig.show()


    # *
    # * Temperature-dependent update
    # *

    lookup_time, lookup_std, numpy_time, numpy_std, scipy_time, scipy_std = compare_efficiency()

    # NumPy interpolation
    numpy_speedup = lookup_time / numpy_time
    speedup_str = f"{numpy_speedup:.1f}x" if numpy_speedup > 1 else f"0.{int(1/numpy_speedup)}x"

    # SciPy cubic interpolation
    scipy_speedup = lookup_time / scipy_time
    speedup_str = f"{scipy_speedup:.1f}x" if scipy_speedup > 1 else f"0.{int(1/scipy_speedup)}x"
    
    # Log
    print(f"{'Method':<20} {'Time (ms)':<12} {'Std (ms)':<12} {'Speedup':<10}")
    print(f"{'Lookup Table':<20} {lookup_time*1000:<12.3f} {lookup_std*1000:<12.3f} {'1.0x':<10}")
    print(f"{'NumPy Interp':<20} {numpy_time*1000:<12.3f} {numpy_std*1000:<12.3f} {speedup_str:<10}")
    print(f"{'SciPy Cubic':<20} {scipy_time*1000:<12.3f} {scipy_std*1000:<12.3f} {speedup_str:<10}")

    # Create visualization
    methods = ['Lookup Table', 'NumPy Interp', 'SciPy Cubic']
    times_ms = [lookup_time * 1000, numpy_time * 1000, scipy_time * 1000]
    stds_ms = [lookup_std * 1000, numpy_std * 1000, scipy_std * 1000]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=methods,
        y=times_ms,
        error_y=dict(type='data', array=stds_ms),
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        text=[f'{t:.3f} ms' for t in times_ms],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Interpolation Methods Performance',
        xaxis_title='Method',
        yaxis_title='Execution Time (ms)',
        #yaxis_type='log',
        template='plotly_white',
        font=dict(size=20),
        width=1000, height=600,
    )

    fig.write_image("src/plots/temperature_update.png", scale=2)
    fig.write_html("src/plots/temperature_update.html")
    fig.show()
