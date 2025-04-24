document.addEventListener('DOMContentLoaded', () => {
    // Initialize charts with better styling and layouts
    const commonLayout = {
        autosize: true,
        margin: { t: 30, l: 50, r: 30, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#333' },
        xaxis: {
            gridcolor: '#eee',
            title: 'Time'
        },
        yaxis: {
            gridcolor: '#eee'
        }
    };

    // CPU Chart
    Plotly.newPlot('cpu-chart', [{
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines+markers',
        name: 'CPU Usage',
        line: { color: '#2196F3', width: 2 },
        marker: { color: '#2196F3', size: 5 }
    }], {
        ...commonLayout,
        title: 'CPU Usage (%)',
        yaxis: { ...commonLayout.yaxis, range: [0, 100] }
    });

    // GPU Chart
    Plotly.newPlot('gpu-chart', [{
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines+markers',
        name: 'GPU Usage',
        line: { color: '#F44336', width: 2 },
        marker: { color: '#F44336', size: 5 }
    }], {
        ...commonLayout,
        title: 'GPU Usage (%)',
        yaxis: { ...commonLayout.yaxis, range: [0, 100] }
    });

    // Memory Chart
    Plotly.newPlot('memory-chart', [{
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Memory Usage',
        line: { color: '#4CAF50', width: 2 },
        marker: { color: '#4CAF50', size: 5 }
    }], {
        ...commonLayout,
        title: 'Memory Usage (%)',
        yaxis: { ...commonLayout.yaxis, range: [0, 100] }
    });

    // Network Chart
    Plotly.newPlot('network-chart', [
        {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Data In (KB)',
            line: { color: '#9C27B0', width: 2 },
            marker: { color: '#9C27B0', size: 5 }
        },
        {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Data Out (KB)',
            line: { color: '#FF9800', width: 2 },
            marker: { color: '#FF9800', size: 5 }
        }
    ], {
        ...commonLayout,
        title: 'Network Traffic (KB)',
        legend: { orientation: 'h', y: 1.1 }
    });

    // Anomaly Chart
    Plotly.newPlot('anomaly-chart', [{
        x: [],
        y: [],
        type: 'scatter',
        mode: 'markers',
        name: 'Anomaly Score',
        marker: {
            color: '#E91E63',
            size: 10,
            symbol: 'circle'
        }
    }], {
        ...commonLayout,
        title: 'Anomaly Detection',
        yaxis: { ...commonLayout.yaxis, range: [-0.1, 1.1], title: 'Anomaly Score (0=Normal, 1=Anomaly)' }
    });

    // Function to update charts with new data
    async function updateCharts() {
        try {
            const response = await fetch('/api/metrics');
            const data = await response.json();
            
            // Update CPU chart
            Plotly.update('cpu-chart', {
                x: [data.timestamps],
                y: [data.cpu]
            }, {});

            // Update GPU chart
            Plotly.update('gpu-chart', {
                x: [data.timestamps],
                y: [data.gpu]
            }, {});

            // Update memory chart
            Plotly.update('memory-chart', {
                x: [data.timestamps],
                y: [data.memory]
            }, {});

            // Update network chart
            Plotly.update('network-chart', {
                x: [data.timestamps, data.timestamps],
                y: [data.data_in, data.data_out]
            }, {});

            // Update anomaly chart - use different colors based on anomaly score
            const colors = data.anomaly_scores.map(score => score === 0 ? '#4CAF50' : '#F44336');
            Plotly.update('anomaly-chart', {
                x: [data.timestamps],
                y: [data.anomaly_scores],
                'marker.color': [colors]
            }, {});

            // Update dashboard summary
            updateSummary(data);
        } catch (error) {
            console.error('Error updating metrics:', error);
        }
    }

    // Function to update the dashboard summary
    function updateSummary(data) {
        const summaryElement = document.getElementById('dashboard-summary');
        if (!summaryElement) return;

        const lastIndex = data.cpu.length - 1;
        const currentCpu = data.cpu[lastIndex];
        const currentGpu = data.gpu[lastIndex];
        const currentMemory = data.memory[lastIndex];
        const currentDataIn = data.data_in[lastIndex];
        const currentDataOut = data.data_out[lastIndex];
        const hasAnomaly = data.anomaly_scores.includes(1);

        let statusClass = 'status-normal';
        let statusMessage = 'System operating normally';

        if (currentCpu > 80 || currentMemory > 80) {
            statusClass = 'status-warning';
            statusMessage = 'High resource usage detected';
        }

        if (hasAnomaly) {
            statusClass = 'status-alert';
            statusMessage = 'Anomaly detected in system metrics';
        }

        summaryElement.innerHTML = `
            <div class="status-indicator ${statusClass}">
                <span>${statusMessage}</span>
            </div>
            <div class="metrics-summary">
                <div class="metric-item">
                    <span class="metric-label">CPU:</span>
                    <span class="metric-value">${currentCpu}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">GPU:</span>
                    <span class="metric-value">${currentGpu}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Memory:</span>
                    <span class="metric-value">${currentMemory}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Network In:</span>
                    <span class="metric-value">${currentDataIn} KB</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Network Out:</span>
                    <span class="metric-value">${currentDataOut} KB</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Last Updated:</span>
                    <span class="metric-value">${new Date().toLocaleTimeString()}</span>
                </div>
            </div>
        `;
    }

    // Fetch and update data periodically
    setInterval(updateCharts, 5000);
    
    // Initial update
    updateCharts();
});