"""Generate Interactive HTML Dashboard for Analysis Results

Creates a self-contained HTML file with embedded JSON data and interactive charts.

NOTE: This is a temporary convenience utility for exploring results.
A full-featured dashboard solution is planned for future releases.
"""

import json
from pathlib import Path

from agentune.analyze.feature.problem import ClassificationProblem
from agentune.analyze.run.analysis.base import AnalyzeResults

# Constants
BOOLEAN_CATEGORY_COUNT = 2  # Number of categories for boolean features


def _looks_like_number_or_bin(category_str: str) -> bool:
    """Check if a category string looks like a number or numeric bin"""
    if not isinstance(category_str, str):
        return False

    # Check for numeric bins like "[1.0, 2.0)", "(-inf, 0.5]", etc.
    if any(char in category_str for char in ['[', ']', '(', ')', 'inf', '-']):
        return True

    # Try to parse as float
    try:
        float(category_str)
        return True
    except ValueError:
        return False


def extract_dashboard_data(results: AnalyzeResults) -> dict:
    """Extract and process data for dashboard from AnalyzeResults"""
    features_with_stats = results.features_with_train_stats

    # Extract feature data
    features = []
    for feature_with_stats in features_with_stats:
        feature = feature_with_stats.feature
        stats = feature_with_stats.stats

        # Determine feature type based on the original feature type
        feature_type = 'categorical'  # default

        # Check if it's a boolean feature (2 true/false values)
        if (len(stats.feature.categories) == BOOLEAN_CATEGORY_COUNT and
            set(stats.feature.categories) <= {'True', 'False', 'true', 'false'}):
            feature_type = 'boolean'
        # Check if it's numeric based on feature type
        elif feature.is_numeric():
            feature_type = 'numeric'
        # Alternative: check if histogram data exists (robust indicator of numeric feature)
        elif hasattr(stats.feature, 'histogram_counts') and hasattr(stats.feature, 'histogram_bin_edges') and \
             getattr(stats.feature, 'histogram_counts', None) and getattr(stats.feature, 'histogram_bin_edges', None):
            feature_type = 'numeric'
        # Fallback: check if all categories look like numbers or numeric bins
        elif all(_looks_like_number_or_bin(cat) for cat in stats.feature.categories[:5]):  # Check first 5
            feature_type = 'numeric'

        feature_data = {
            'name': feature.name,
            'description': feature.description,
            'technical_description': feature.technical_description,
            'default_for_missing': feature.default_for_missing,
            'r_squared': stats.relationship.r_squared,
            'sse_reduction': stats.relationship.sse_reduction,  # Keep for reference
            'n_total': stats.feature.n_total,
            'n_missing': stats.feature.n_missing,
            'missing_percentage': (stats.feature.n_missing / stats.feature.n_total) * 100 if stats.feature.n_total > 0 else 0,
            'categories': list(stats.feature.categories),
            'value_counts': dict(stats.feature.value_counts),
            'target_classes': list(stats.relationship.classes),
            'lift_matrix': [list(row) for row in stats.relationship.lift],
            'mean_shift': [list(row) for row in stats.relationship.mean_shift],
            'support': list(stats.feature.support),
            'histogram_counts': list(getattr(stats.feature, 'histogram_counts', [])),
            'histogram_bin_edges': list(getattr(stats.feature, 'histogram_bin_edges', [])),
            'feature_type': feature_type,
            # Keep is_boolean for backward compatibility
            'is_boolean': feature_type == 'boolean'
        }
        features.append(feature_data)

    # Sort features by R² in descending order (highest R² first)
    features.sort(key=lambda f: f['r_squared'], reverse=True)

    # Calculate summary statistics
    r_squared_values = [f['r_squared'] for f in features]
    missing_rates = [f['missing_percentage'] for f in features]

    # Get target distribution and desired outcome
    target_distribution = {}
    desired_outcome = None
    if features_with_stats and isinstance(results.problem, ClassificationProblem):
        target_classes = results.problem.classes
        totals_per_class = features_with_stats[0].stats.relationship.totals_per_class
        target_distribution = dict(zip(target_classes, totals_per_class, strict=False))
        # Extract desired outcome from problem description
        if hasattr(results.problem, 'problem_description') and results.problem.problem_description:
            desired_outcome = results.problem.problem_description.target_desired_outcome_value

    # Extract actual dataset size from first feature (if available)
    actual_dataset_rows = None
    if features:
        actual_dataset_rows = features[0]['n_total']

    summary = {
        'total_features': len(features),
        'mean_r_squared': sum(r_squared_values) / len(r_squared_values) if r_squared_values else 0,
        'max_r_squared': max(r_squared_values) if r_squared_values else 0,
        'min_r_squared': min(r_squared_values) if r_squared_values else 0,
        'mean_missing': sum(missing_rates) / len(missing_rates) if missing_rates else 0,
        'boolean_features': sum(1 for f in features if f['feature_type'] == 'boolean'),
        'numeric_features': sum(1 for f in features if f['feature_type'] == 'numeric'),
        'categorical_features': sum(1 for f in features if f['feature_type'] == 'categorical'),
        'target_classes': list(target_distribution.keys()),
        'target_distribution': target_distribution,
        'desired_outcome': desired_outcome,
        'dataset_rows_actual': actual_dataset_rows
    }

    return {
        'features': features,
        'summary': summary
    }


def create_analyze_dashboard(
    results: AnalyzeResults,
    output_file: str | Path | None = None,
    title: str = 'Analysis Results Dashboard'
) -> Path:
    """Generate interactive HTML dashboard from AnalyzeResults.

    NOTE: This is a temporary convenience utility for exploring results.
    A full-featured dashboard solution is planned for future releases.

    Args:
        results: Analysis results from ctx.ops.analyze()
        output_file: Optional path to save HTML file (default: "feature_dashboard.html")
        title: Dashboard title

    Returns:
        Path to the generated HTML file
    """
    # Extract data from results
    data = extract_dashboard_data(results)

    # Generate HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8fafc;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background: white;
            padding: 18px 12px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s ease;
            min-height: 95px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
        }}

        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #4f46e5;
            margin-bottom: 8px;
            line-height: 1;
        }}

        .stat-label {{
            color: #6b7280;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            line-height: 1.2;
        }}

        .charts-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}

        .chart-container {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}

        .chart-title {{
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #374151;
        }}

        .section {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 30px;
            overflow: hidden;
        }}

        .section-header {{
            padding: 25px;
            border-bottom: 1px solid #e5e7eb;
            background: #f9fafb;
        }}

        .section-title {{
            font-size: 1.5em;
            font-weight: 600;
            color: #374151;
        }}

        .feature-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .feature-table th,
        .feature-table td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}

        .feature-table th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
            cursor: pointer;
            user-select: none;
            position: relative;
        }}

        .feature-table th:hover {{
            background: #f3f4f6;
        }}

        .feature-table tr {{
            cursor: pointer;
            transition: background-color 0.2s ease;
        }}

        .feature-table tr:hover {{
            background: #f9fafb;
        }}

        .r2-high {{ color: #059669; font-weight: bold; }}
        .r2-medium {{ color: #d97706; font-weight: bold; }}
        .r2-low {{ color: #dc2626; font-weight: bold; }}

        .feature-details {{
            display: none;
            padding: 25px;
            background: #f8fafc;
            border-top: 1px solid #e5e7eb;
        }}

        .feature-details.show {{
            display: block;
            animation: fadeIn 0.3s ease;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .detail-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}

        .detail-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .detail-title {{
            font-weight: 600;
            margin-bottom: 15px;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 8px;
        }}

        .comparison-section {{
            padding: 25px;
        }}

        .comparison-controls {{
            margin-bottom: 20px;
        }}

        .feature-checkbox {{
            display: inline-flex;
            align-items: center;
            margin: 5px 10px 5px 0;
            padding: 8px 12px;
            background: #f3f4f6;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }}

        .feature-checkbox:hover {{
            background: #e5e7eb;
        }}

        .feature-checkbox input {{
            margin-right: 8px;
        }}

        .comparison-results {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}

        .comparison-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .sort-arrow {{
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            opacity: 0.5;
        }}

        .tabs {{
            display: flex;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 15px;
        }}

        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 0.9em;
            color: #6b7280;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}

        .tab.active {{
            color: #0969da;
            border-bottom-color: #0969da;
        }}

        .tab:hover {{
            color: #0969da;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .toggle-group {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
        }}

        .toggle-button {{
            padding: 8px 16px;
            border: 1px solid #d1d9e0;
            background: white;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.2s;
        }}

        .toggle-button:hover {{
            background: #f6f8fa;
        }}

        .toggle-button.active {{
            background: #0969da;
            color: white;
            border-color: #0969da;
        }}

        @media (max-width: 768px) {{
            .charts-section {{
                grid-template-columns: 1fr;
            }}

            .detail-grid {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{title}</h1>
            <p>Interactive dashboard for analyzing generated features and their performance</p>
        </div>


        <!-- Summary Statistics -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{data['summary']['total_features']}</div>
                <div class="stat-label">Total Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data['summary']['max_r_squared']:.4f}</div>
                <div class="stat-label">Best R²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data['summary']['mean_r_squared']:.4f}</div>
                <div class="stat-label">Average R²</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{data['summary']['mean_missing']:.1f}%</div>
                <div class="stat-label">Avg Missing Data</div>
            </div>
            {f'''<div class="stat-card">
                <div class="stat-value">{data['summary']['boolean_features']}</div>
                <div class="stat-label">Boolean Features</div>
            </div>''' if data['summary']['boolean_features'] > 0 else ''}
            {f'''<div class="stat-card">
                <div class="stat-value">{data['summary']['numeric_features']}</div>
                <div class="stat-label">Numeric Features</div>
            </div>''' if data['summary']['numeric_features'] > 0 else ''}
            {f'''<div class="stat-card">
                <div class="stat-value">{data['summary']['categorical_features']}</div>
                <div class="stat-label">Categorical Features</div>
            </div>''' if data['summary']['categorical_features'] > 0 else ''}
            {f'''<div class="stat-card">
                <div class="stat-value">{data['summary']['dataset_rows_actual']}</div>
                <div class="stat-label">Dataset Rows</div>
            </div>''' if data['summary'].get('dataset_rows_actual') else ''}
        </div>

        <!-- Charts Section -->
        <div class="charts-section">
            <div class="chart-container">
                <div class="chart-title">Target Class Distribution</div>
                <canvas id="targetChart" width="400" height="300"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">R² (Coefficient of Determination) Ranking</div>
                <canvas id="r2Chart" width="400" height="300"></canvas>
            </div>
        </div>

        <!-- Feature Table -->
        <div class="section">
            <div class="section-header">
                <div class="section-title">Feature Performance Table</div>
            </div>
            <table class="feature-table" id="featureTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">Feature Name <span class="sort-arrow">↕</span></th>
                        <th onclick="sortTable(1)">R² <span class="sort-arrow">↕</span></th>
                        <th onclick="sortTable(2)">Missing % <span class="sort-arrow">↕</span></th>
                        <th onclick="sortTable(3)">Categories <span class="sort-arrow">↕</span></th>
                        <th onclick="sortTable(4)">Type <span class="sort-arrow">↕</span></th>
                    </tr>
                </thead>
                <tbody id="featureTableBody">
                    <!-- Table rows will be populated by JavaScript -->
                </tbody>
            </table>
        </div>

        <!-- Feature Comparison -->
        <div class="section">
            <div class="section-header">
                <div class="section-title">Feature Comparison</div>
            </div>
            <div class="comparison-section">
                <div class="comparison-controls">
                    <div style="margin-bottom: 15px; font-weight: 500;">Select features to compare:</div>
                    <div id="comparisonControls">
                        <!-- Checkboxes will be populated by JavaScript -->
                    </div>
                    <div class="toggle-group" id="comparisonToggle" style="display: none;">
                        <span style="font-weight: 500;">View:</span>
                        <button class="toggle-button active" onclick="toggleComparisonView('shift')">Mean Shift</button>
                        <button class="toggle-button" onclick="toggleComparisonView('lift')">Lift Analysis</button>
                    </div>
                </div>
                <div id="comparisonResults" class="comparison-results">
                    <!-- Comparison results will be shown here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // Embedded data
        const dashboardData = {json.dumps(data, indent=2)};

        // Global variables
        let currentSort = {{ column: null, ascending: false }}; // Start with no sorting (preserve original order)
        let selectedFeatures = new Set();
        let comparisonViewType = 'shift';

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {{
            initializeCharts();
            populateFeatureTable();
            setupComparisonControls();
        }});

        // Initialize charts
        function initializeCharts() {{
            // Target distribution pie chart
            const targetCtx = document.getElementById('targetChart').getContext('2d');
            const targetData = dashboardData.summary.target_distribution;
            const labels = Object.keys(targetData);
            const desiredOutcome = dashboardData.summary.desired_outcome;

            // Color palette - supports up to 12 classes
            const colorPalette = [
                '#3b82f6',  // Blue
                '#22c55e',  // Green
                '#f59e0b',  // Amber
                '#ef4444',  // Red
                '#a855f7',  // Purple
                '#06b6d4',  // Cyan
                '#f97316',  // Orange
                '#ec4899',  // Pink
                '#84cc16',  // Lime
                '#14b8a6',  // Teal
                '#f43f5e',  // Rose
                '#8b5cf6'   // Violet
            ];

            // Assign colors and borders - highlight desired outcome with gold border
            const backgroundColors = [];
            const borderColors = [];
            const borderWidths = [];

            labels.forEach((label, index) => {{
                backgroundColors.push(colorPalette[index % colorPalette.length]);

                // Highlight desired outcome with thick gold border
                if (desiredOutcome && label === desiredOutcome) {{
                    borderColors.push('#fbbf24');  // Gold
                    borderWidths.push(6);
                }} else {{
                    borderColors.push('#ffffff');
                    borderWidths.push(2);
                }}
            }});

            new Chart(targetCtx, {{
                type: 'pie',
                data: {{
                    labels: labels,
                    datasets: [{{
                        data: Object.values(targetData),
                        backgroundColor: backgroundColors,
                        borderWidth: borderWidths,
                        borderColor: borderColors
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                padding: 20,
                                usePointStyle: true
                            }}
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                    const percentage = ((context.parsed / total) * 100).toFixed(1);
                                    return `${{context.label}}: ${{context.parsed}} (${{percentage}}%)`;
                                }}
                            }}
                        }}
                    }}
                }}
            }});

            // R² horizontal bar chart
            const r2Ctx = document.getElementById('r2Chart').getContext('2d');
            // Sort by R² (highest first) and take top 8
            const features = [...dashboardData.features]
                .sort((a, b) => b.r_squared - a.r_squared)
                .slice(0, 8);

            new Chart(r2Ctx, {{
                type: 'bar',
                data: {{
                    labels: features.map(f => f.name.length > 20 ? f.name.substring(0, 20) + '...' : f.name),
                    datasets: [{{
                        label: 'R² (Coefficient of Determination)',
                        data: features.map(f => f.r_squared),
                        backgroundColor: features.map(f =>
                            f.r_squared >= 0.01 ? '#22c55e' :
                            f.r_squared >= 0.005 ? '#f59e0b' : '#ef4444'
                        ),
                        borderRadius: 4,
                        borderSkipped: false
                    }}]
                }},
                options: {{
                    indexAxis: 'y',
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        tooltip: {{
                            callbacks: {{
                                title: function(context) {{
                                    return features[context[0].dataIndex].name;
                                }},
                                label: function(context) {{
                                    return `R²: ${{context.parsed.x.toFixed(4)}}`;
                                }},
                                afterLabel: function(context) {{
                                    const feature = features[context.dataIndex];
                                    return feature.description;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'R² (Coefficient of Determination)'
                            }}
                        }}
                    }}
                }}
            }});
        }}

        // Populate feature table
        function populateFeatureTable() {{
            const tbody = document.getElementById('featureTableBody');
            tbody.innerHTML = '';

            // Use original order by default (preserve feature selector ranking)
            let featuresToShow = dashboardData.features;

            // Only sort if user has clicked a column header
            if (currentSort.column !== null) {{
                featuresToShow = [...dashboardData.features].sort((a, b) => {{
                    let aValue, bValue;

                    switch(currentSort.column) {{
                        case 0: // Name
                            aValue = a.name.toLowerCase();
                            bValue = b.name.toLowerCase();
                            break;
                        case 1: // R²
                            aValue = a.r_squared;
                            bValue = b.r_squared;
                            break;
                        case 2: // Missing %
                            aValue = a.missing_percentage;
                            bValue = b.missing_percentage;
                            break;
                        case 3: // Categories
                            aValue = a.categories.length;
                            bValue = b.categories.length;
                            break;
                        case 4: // Type
                            aValue = a.feature_type.charAt(0).toUpperCase() + a.feature_type.slice(1);
                            bValue = b.feature_type.charAt(0).toUpperCase() + b.feature_type.slice(1);
                            break;
                        default:
                            return 0; // No sorting
                    }}

                    if (typeof aValue === 'string') {{
                        return currentSort.ascending ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
                    }} else {{
                        return currentSort.ascending ? aValue - bValue : bValue - aValue;
                    }}
                }});
            }}

            featuresToShow.forEach((feature, index) => {{
                const row = document.createElement('tr');
                row.onclick = () => toggleFeatureDetails(index, feature);

                const r2Class = feature.r_squared >= 0.01 ? 'r2-high' :
                               feature.r_squared >= 0.005 ? 'r2-medium' : 'r2-low';

                row.innerHTML = `
                    <td title="${{feature.description}}" style="cursor: help;">${{feature.name}}</td>
                    <td class="${{r2Class}}">${{feature.r_squared.toFixed(4)}}</td>
                    <td>${{feature.missing_percentage.toFixed(1)}}%</td>
                    <td>${{feature.categories.length}}</td>
                    <td>${{feature.feature_type.charAt(0).toUpperCase() + feature.feature_type.slice(1)}}</td>
                `;

                tbody.appendChild(row);

                // Add details row (initially hidden)
                const detailsRow = document.createElement('tr');
                detailsRow.innerHTML = `
                    <td colspan="5">
                        <div class="feature-details" id="details-${{index}}">
                            <div class="detail-grid">
                                <div class="detail-section">
                                    <div class="detail-title">Description</div>
                                    <p>${{feature.description}}</p>
                                    ${{feature.technical_description ? `
                                        <div class="detail-title" style="margin-top: 15px;">Technical Details</div>
                                        <div style="background: #f8fafc; padding: 12px; border-radius: 6px; font-size: 0.9em; color: #4b5563; border-left: 3px solid #3b82f6;">
                                            ${{feature.technical_description}}
                                        </div>
                                    ` : ''}}
                                    ${{feature.default_for_missing !== undefined && feature.default_for_missing !== null ? `
                                        <p style="margin-top: 10px;"><strong>Default for Missing:</strong>
                                        <code style="background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 0.9em;">${{feature.default_for_missing}}</code></p>
                                    ` : ''}}
                                    <div class="detail-title" style="margin-top: 20px;">Value Distribution</div>
                                    <canvas id="distribution-${{index}}" width="400" height="200"></canvas>
                                </div>
                                <div class="detail-section">
                                    <div class="detail-title">Performance Metrics</div>
                                    <p><strong>R² (Coefficient of Determination):</strong> ${{feature.r_squared.toFixed(4)}}</p>
                                    <p><strong>Sample Size:</strong> ${{feature.n_total}}</p>
                                    <p><strong>Missing Values:</strong> ${{feature.n_missing}} (${{feature.missing_percentage.toFixed(1)}}%)</p>

                                    <div class="tabs" style="margin-top: 20px;">
                                        <button class="tab active" onclick="switchTab(${{index}}, 'lift')">Lift Analysis</button>
                                        <button class="tab" onclick="switchTab(${{index}}, 'shift')">Mean Shift</button>
                                        ${{feature.histogram_counts && feature.histogram_counts.length > 0 ?
                                            `<button class="tab" onclick="switchTab(${{index}}, 'histogram')">Histogram</button>` : ''}}
                                    </div>

                                    <div id="lift-tab-${{index}}" class="tab-content active">
                                        <canvas id="lift-${{index}}" width="400" height="200"></canvas>
                                    </div>

                                    <div id="shift-tab-${{index}}" class="tab-content">
                                        <div id="shift-content-${{index}}"></div>
                                    </div>

                                    ${{feature.histogram_counts && feature.histogram_counts.length > 0 ?
                                        `<div id="histogram-tab-${{index}}" class="tab-content">
                                            <canvas id="histogram-${{index}}" width="400" height="200"></canvas>
                                        </div>` : ''}}
                                </div>
                            </div>
                        </div>
                    </td>
                `;
                tbody.appendChild(detailsRow);
            }});
        }}

        // Toggle feature details
        function toggleFeatureDetails(index, feature) {{
            const details = document.getElementById(`details-${{index}}`);
            const isVisible = details.classList.contains('show');

            // Hide all other details first
            document.querySelectorAll('.feature-details').forEach(detail => {{
                detail.classList.remove('show');
            }});

            if (!isVisible) {{
                details.classList.add('show');

                // Create charts
                setTimeout(() => {{
                    createDistributionChart(index, feature);
                    createLiftMatrix(index, feature);
                    createMeanShiftContent(index, feature);
                    // Create histogram chart if data is available
                    if (feature.histogram_counts && feature.histogram_counts.length > 0) {{
                        createHistogramChart(index, feature);
                    }}
                }}, 100);
            }}
        }}

        // Create distribution chart for a feature
        function createDistributionChart(index, feature) {{
            const ctx = document.getElementById(`distribution-${{index}}`);
            if (!ctx) return;

            const chartInstance = Chart.getChart(ctx);
            if (chartInstance) {{
                chartInstance.destroy();
            }}

            new Chart(ctx.getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: feature.categories,
                    datasets: [{{
                        label: 'Count',
                        data: feature.categories.map(cat => feature.value_counts[cat] || 0),
                        backgroundColor: '#3b82f6',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }}

        // Create lift matrix heatmap table
        function createLiftMatrix(index, feature) {{
            const container = document.getElementById(`lift-${{index}}`).parentElement;

            // Remove the canvas and create a table instead
            const canvas = document.getElementById(`lift-${{index}}`);
            canvas.remove();

            const table = document.createElement('table');
            table.style.cssText = 'width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 10px;';

            // Create header row
            const headerRow = document.createElement('tr');
            headerRow.innerHTML = '<th style="border: 1px solid #e5e7eb; padding: 8px; background: #f9fafb;">Feature → Target</th>' +
                feature.target_classes.map(targetClass =>
                    `<th style="border: 1px solid #e5e7eb; padding: 8px; background: #f9fafb; text-align: center;">${{targetClass}}</th>`
                ).join('');
            table.appendChild(headerRow);

            // Create data rows
            feature.categories.forEach((category, i) => {{
                const row = document.createElement('tr');
                let rowHtml = `<td style="border: 1px solid #e5e7eb; padding: 8px; font-weight: 600;">${{category}}</td>`;

                feature.lift_matrix[i].forEach(liftValue => {{
                    let backgroundColor = '#f9fafb';
                    let textColor = '#374151';

                    // Color coding based on lift value (>1 = green/good, <1 = red/bad)
                    if (liftValue > 1.5) {{
                        backgroundColor = '#dcfce7'; // Green for high lift (good)
                        textColor = '#166534';
                    }} else if (liftValue > 1.2) {{
                        backgroundColor = '#f0fdf4'; // Light green for medium-high lift
                        textColor = '#15803d';
                    }} else if (liftValue > 1.0) {{
                        backgroundColor = '#f7fee7'; // Very light green for medium lift
                        textColor = '#365314';
                    }} else if (liftValue > 0.8) {{
                        backgroundColor = '#fef7f7'; // Light red for low lift (bad)
                        textColor = '#7f1d1d';
                    }} else {{
                        backgroundColor = '#fee2e2'; // Red for very low lift (bad)
                        textColor = '#dc2626';
                    }}

                    rowHtml += `<td style="border: 1px solid #e5e7eb; padding: 8px; text-align: center; background-color: ${{backgroundColor}}; color: ${{textColor}}; font-weight: 600;">${{liftValue.toFixed(2)}}</td>`;
                }});

                row.innerHTML = rowHtml;
                table.appendChild(row);
            }});

            // Add legend
            const legend = document.createElement('div');
            legend.style.cssText = 'margin-top: 10px; font-size: 0.8em; color: #6b7280;';
            legend.innerHTML = `
                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #dcfce7; margin-right: 4px;"></span>High (>1.5)</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #f0fdf4; margin-right: 4px;"></span>Med-High (>1.2)</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #f7fee7; margin-right: 4px;"></span>Medium (>1.0)</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #fef7f7; margin-right: 4px;"></span>Low (>0.8)</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #fee2e2; margin-right: 4px;"></span>Very Low (≤0.8)</span>
                </div>
            `;

            container.appendChild(table);
            container.appendChild(legend);
        }}

        // Create mean shift content
        function createMeanShiftContent(index, feature) {{
            const container = document.getElementById(`shift-content-${{index}}`);
            if (!container || !feature.mean_shift || !feature.mean_shift.length) return;

            const table = document.createElement('table');
            table.style.cssText = 'width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 10px;';

            // Create header row
            const headerRow = document.createElement('tr');
            headerRow.innerHTML = '<th style="border: 1px solid #e5e7eb; padding: 8px; background: #f9fafb;">Feature → Target</th>' +
                feature.target_classes.map(targetClass =>
                    `<th style="border: 1px solid #e5e7eb; padding: 8px; background: #f9fafb; text-align: center;">${{targetClass}}</th>`
                ).join('');
            table.appendChild(headerRow);

            // Create data rows
            feature.categories.forEach((category, i) => {{
                const row = document.createElement('tr');
                let rowHtml = `<td style="border: 1px solid #e5e7eb; padding: 8px; font-weight: 600;">${{category}}</td>`;

                feature.mean_shift[i].forEach(shiftValue => {{
                    let backgroundColor = '#f9fafb';
                    let textColor = '#374151';
                    const absShift = Math.abs(shiftValue);

                    // Color coding based on mean shift value
                    if (absShift > 0.1) {{
                        backgroundColor = shiftValue > 0 ? '#dcfce7' : '#fee2e2';
                        textColor = shiftValue > 0 ? '#166534' : '#dc2626';
                    }} else if (absShift > 0.05) {{
                        backgroundColor = shiftValue > 0 ? '#f0fdf4' : '#fef2f2';
                        textColor = shiftValue > 0 ? '#15803d' : '#e11d48';
                    }} else if (absShift > 0.02) {{
                        backgroundColor = shiftValue > 0 ? '#f7fee7' : '#fef7f7';
                        textColor = shiftValue > 0 ? '#365314' : '#7f1d1d';
                    }}

                    rowHtml += `<td style="border: 1px solid #e5e7eb; padding: 8px; text-align: center; background-color: ${{backgroundColor}}; color: ${{textColor}}; font-weight: 600;">${{shiftValue.toFixed(3)}}</td>`;
                }});

                row.innerHTML = rowHtml;
                table.appendChild(row);
            }});

            // Add legend
            const legend = document.createElement('div');
            legend.style.cssText = 'margin-top: 10px; font-size: 0.8em; color: #6b7280;';
            legend.innerHTML = `
                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #dcfce7; margin-right: 4px;"></span>High Positive (>0.1)</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #fee2e2; margin-right: 4px;"></span>High Negative (<-0.1)</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #f0fdf4; margin-right: 4px;"></span>Med Positive (>0.05)</span>
                    <span><span style="display: inline-block; width: 12px; height: 12px; background: #fef2f2; margin-right: 4px;"></span>Med Negative (<-0.05)</span>
                </div>
            `;

            container.appendChild(table);
            container.appendChild(legend);
        }}

        // Create histogram chart for numeric features
        function createHistogramChart(index, feature) {{
            const ctx = document.getElementById(`histogram-${{index}}`);
            if (!ctx || !feature.histogram_counts || !feature.histogram_bin_edges) return;

            const chartInstance = Chart.getChart(ctx);
            if (chartInstance) {{
                chartInstance.destroy();
            }}

            // Convert bin edges to range labels
            const binLabels = [];
            for (let i = 0; i < feature.histogram_bin_edges.length - 1; i++) {{
                const start = feature.histogram_bin_edges[i].toFixed(1);
                const end = feature.histogram_bin_edges[i + 1].toFixed(1);
                binLabels.push(`[${{start}} - ${{end}})`);
            }}

            new Chart(ctx.getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels: binLabels,
                    datasets: [{{
                        label: 'Count',
                        data: feature.histogram_counts,
                        backgroundColor: '#3b82f6',
                        borderColor: '#1e40af',
                        borderWidth: 1,
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        tooltip: {{
                            callbacks: {{
                                title: function(context) {{
                                    return `Range: ${{context[0].label}}`;
                                }},
                                label: function(context) {{
                                    return `Count: ${{context.parsed.y}}`;
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Count'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Value Range'
                            }},
                            ticks: {{
                                maxRotation: 45
                            }}
                        }}
                    }}
                }}
            }});
        }}

        // Switch tabs in feature details
        function switchTab(index, tabType) {{
            // Update tab buttons - handle variable number of tabs
            const tabsContainer = document.querySelector(`#details-${{index}} .tabs`);
            const allTabs = tabsContainer.querySelectorAll('.tab');

            allTabs.forEach(tab => {{
                const isActive =
                    (tabType === 'lift' && tab.textContent === 'Lift Analysis') ||
                    (tabType === 'shift' && tab.textContent === 'Mean Shift') ||
                    (tabType === 'histogram' && tab.textContent === 'Histogram');
                tab.classList.toggle('active', isActive);
            }});

            // Update tab content
            const liftContent = document.getElementById(`lift-tab-${{index}}`);
            const shiftContent = document.getElementById(`shift-tab-${{index}}`);
            const histogramContent = document.getElementById(`histogram-tab-${{index}}`);

            if (liftContent) liftContent.classList.toggle('active', tabType === 'lift');
            if (shiftContent) shiftContent.classList.toggle('active', tabType === 'shift');
            if (histogramContent) histogramContent.classList.toggle('active', tabType === 'histogram');
        }}

        // Create comparison stats based on view type
        function createComparisonStats(feature, viewType) {{
            if (viewType === 'shift' && feature.mean_shift && feature.mean_shift.length > 0) {{
                return `
                    <div style="margin-top: 15px;">
                        <h5 style="margin-bottom: 10px; color: #0969da;">Target Impact (Mean Shift)</h5>
                        <table style="width: 100%; font-size: 0.85em; border-collapse: collapse;">
                            <thead>
                                <tr>
                                    <th style="text-align: left; padding: 6px; border-bottom: 1px solid #e5e7eb;">Category</th>
                                    ${{feature.target_classes.map(tc => `<th style="text-align: center; padding: 6px; border-bottom: 1px solid #e5e7eb;">${{tc}}</th>`).join('')}}
                                </tr>
                            </thead>
                            <tbody>
                                ${{feature.categories.map((cat, catIdx) => `
                                    <tr>
                                        <td style="padding: 6px; font-weight: 600;">${{cat}}</td>
                                        ${{feature.mean_shift[catIdx].map(shift => {{
                                            const absShift = Math.abs(shift);
                                            let bgColor = '#f9fafb';
                                            let textColor = '#374151';
                                            if (absShift > 0.1) {{
                                                bgColor = shift > 0 ? '#dcfce7' : '#fee2e2';
                                                textColor = shift > 0 ? '#166534' : '#dc2626';
                                            }} else if (absShift > 0.05) {{
                                                bgColor = shift > 0 ? '#f0fdf4' : '#fef2f2';
                                                textColor = shift > 0 ? '#15803d' : '#e11d48';
                                            }}
                                            return `<td style="padding: 6px; text-align: center; background: ${{bgColor}}; color: ${{textColor}}; font-weight: 600;">${{shift.toFixed(3)}}</td>`;
                                        }}).join('')}}
                                    </tr>
                                `).join('')}}
                            </tbody>
                        </table>
                    </div>
                `;
            }} else if (viewType === 'lift' && feature.lift_matrix && feature.lift_matrix.length > 0) {{
                return `
                    <div style="margin-top: 15px;">
                        <h5 style="margin-bottom: 10px; color: #0969da;">Target Impact (Lift Matrix)</h5>
                        <table style="width: 100%; font-size: 0.85em; border-collapse: collapse;">
                            <thead>
                                <tr>
                                    <th style="text-align: left; padding: 6px; border-bottom: 1px solid #e5e7eb;">Category</th>
                                    ${{feature.target_classes.map(tc => `<th style="text-align: center; padding: 6px; border-bottom: 1px solid #e5e7eb;">${{tc}}</th>`).join('')}}
                                </tr>
                            </thead>
                            <tbody>
                                ${{feature.categories.map((cat, catIdx) => `
                                    <tr>
                                        <td style="padding: 6px; font-weight: 600;">${{cat}}</td>
                                        ${{feature.lift_matrix[catIdx].map(liftValue => {{
                                            let bgColor = '#f9fafb';
                                            let textColor = '#374151';
                                            if (liftValue > 1.5) {{
                                                bgColor = '#dcfce7';
                                                textColor = '#166534';
                                            }} else if (liftValue > 1.2) {{
                                                bgColor = '#f0fdf4';
                                                textColor = '#15803d';
                                            }} else if (liftValue > 1.0) {{
                                                bgColor = '#f7fee7';
                                                textColor = '#365314';
                                            }} else if (liftValue > 0.8) {{
                                                bgColor = '#fef7f7';
                                                textColor = '#7f1d1d';
                                            }} else {{
                                                bgColor = '#fee2e2';
                                                textColor = '#dc2626';
                                            }}
                                            return `<td style="padding: 6px; text-align: center; background: ${{bgColor}}; color: ${{textColor}}; font-weight: 600;">${{liftValue.toFixed(2)}}</td>`;
                                        }}).join('')}}
                                    </tr>
                                `).join('')}}
                            </tbody>
                        </table>
                    </div>
                `;
            }}
            return '';
        }}

        // Toggle comparison view between shift and lift
        function toggleComparisonView(viewType) {{
            comparisonViewType = viewType;

            // Update button states
            const buttons = document.querySelectorAll('#comparisonToggle .toggle-button');
            buttons.forEach(btn => {{
                btn.classList.toggle('active',
                    (viewType === 'shift' && btn.textContent === 'Mean Shift') ||
                    (viewType === 'lift' && btn.textContent === 'Lift Analysis')
                );
            }});

            // Refresh the comparison display
            displayComparison();
        }}

        // Sort table
        function sortTable(columnIndex) {{
            if (currentSort.column === columnIndex) {{
                currentSort.ascending = !currentSort.ascending;
            }} else {{
                currentSort.column = columnIndex;
                currentSort.ascending = false;
            }}
            populateFeatureTable();
        }}

        // Setup comparison controls
        function setupComparisonControls() {{
            const controls = document.getElementById('comparisonControls');

            dashboardData.features.forEach((feature, index) => {{
                const checkbox = document.createElement('label');
                checkbox.className = 'feature-checkbox';
                checkbox.innerHTML = `
                    <input type="checkbox" value="${{index}}" onchange="updateComparison()">
                    <span>${{feature.name}}</span>
                `;
                controls.appendChild(checkbox);
            }});
        }}

        // Update comparison
        function updateComparison() {{
            const checkboxes = document.querySelectorAll('#comparisonControls input[type="checkbox"]');
            selectedFeatures.clear();

            checkboxes.forEach(checkbox => {{
                if (checkbox.checked) {{
                    selectedFeatures.add(parseInt(checkbox.value));
                }}
            }});

            displayComparison();
        }}

        // Display comparison results
        function displayComparison() {{
            const results = document.getElementById('comparisonResults');
            results.innerHTML = '';

            const toggleDiv = document.getElementById('comparisonToggle');

            if (selectedFeatures.size === 0) {{
                results.innerHTML = '<p style="text-align: center; color: #6b7280; padding: 40px;">Select features above to compare them</p>';
                toggleDiv.style.display = 'none';
                return;
            }}

            // Show toggle controls when features are selected
            toggleDiv.style.display = 'flex';

            selectedFeatures.forEach(index => {{
                const feature = dashboardData.features[index];
                const card = document.createElement('div');
                card.className = 'comparison-card';

                const r2Class = feature.r_squared >= 0.01 ? 'r2-high' :
                               feature.r_squared >= 0.005 ? 'r2-medium' : 'r2-low';

                // Create dynamic stats content based on view type
                const targetStatsHtml = createComparisonStats(feature, comparisonViewType);

                card.innerHTML = `
                    <h4>${{feature.name}}</h4>
                    <p style="font-style: italic; color: #6b7280; margin-bottom: 10px;">${{feature.description}}</p>
                    <p><strong>R²:</strong> <span class="${{r2Class}}">${{feature.r_squared.toFixed(4)}}</span></p>
                    <p><strong>Missing:</strong> ${{feature.missing_percentage.toFixed(1)}}%</p>
                    <p><strong>Categories:</strong> ${{feature.categories.length}}</p>
                    <p><strong>Type:</strong> ${{feature.feature_type.charAt(0).toUpperCase() + feature.feature_type.slice(1)}}</p>
                    ${{targetStatsHtml}}
                `;

                results.appendChild(card);
            }});
        }}

    </script>
</body>
</html>'''

    # Determine output path
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path('feature_dashboard.html')

    # Save to file
    with output_path.open('w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path
