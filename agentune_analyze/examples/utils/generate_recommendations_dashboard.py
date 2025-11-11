"""Generate Interactive HTML Dashboard for Recommendations Report

Creates a self-contained HTML file with embedded JSON data and interactive visualizations
for action recommendations.

NOTE: This is a temporary convenience utility for exploring results.
A full-featured dashboard solution is planned for future releases.
"""

from pathlib import Path

from agentune.analyze.feature.recommend.action_recommender import RecommendationsReport


def _escape_html(text: str) -> str:
    """Escape HTML special characters"""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def create_recommendations_dashboard(
    report: RecommendationsReport,
    output_file: str | Path | None = None,
    title: str = 'Action Recommendations Dashboard',
) -> Path:
    """Generate interactive HTML dashboard from RecommendationsReport.

    NOTE: This is a temporary convenience utility for exploring results.
    A full-featured dashboard solution is planned for future releases.

    Args:
        report: Recommendations report from ctx.ops.recommend_conversation_actions()
        output_file: Optional path to save HTML file (default: "recommendations_dashboard.html")
        title: Dashboard title

    Returns:
        Path to the generated HTML file
    """
    # Set default output file
    if output_file is None:
        output_file = Path('recommendations_dashboard.html')
    else:
        output_file = Path(output_file)

    # Get all features from the report (already sorted by R²)
    features_list = [
        f'{feat.name}: {feat.r_squared:.4f}'
        for feat in report.all_features
    ]

    # Generate HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
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
            margin-bottom: 30px;
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
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
        }}

        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #4f46e5;
            margin-bottom: 10px;
        }}

        .stat-label {{
            color: #6b7280;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .section {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 30px;
        }}

        .section h2 {{
            color: #1f2937;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e5e7eb;
        }}

        .analysis-summary {{
            background: #f9fafb;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4f46e5;
            line-height: 1.8;
            white-space: pre-wrap;
        }}

        .recommendation-card {{
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 20px;
            transition: box-shadow 0.2s ease;
        }}

        .recommendation-card:hover {{
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .rec-header {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f3f4f6;
        }}

        .rec-header h3 {{
            color: #4f46e5;
            font-size: 1.4em;
        }}

        .rec-section {{
            margin-bottom: 15px;
        }}

        .rec-section h4 {{
            color: #6b7280;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .rec-section p {{
            color: #374151;
            line-height: 1.7;
            white-space: pre-wrap;
        }}

        .feature-list, .conversation-list {{
            list-style: none;
            padding-left: 0;
        }}

        .feature-list li {{
            padding: 10px;
            margin-bottom: 12px;
            background: #f9fafb;
            border-radius: 6px;
            border-left: 3px solid #e5e7eb;
        }}

        .conversation-list li {{
            padding: 10px;
            margin-bottom: 12px;
            background: #f9fafb;
            border-radius: 6px;
            border-left: 3px solid #e5e7eb;
        }}

        .conv-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }}

        .conv-toggle {{
            background: #4f46e5;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 0.85em;
            cursor: pointer;
            transition: background 0.2s ease;
            white-space: nowrap;
        }}

        .conv-toggle:hover {{
            background: #4338ca;
        }}

        .conv-dropdown {{
            display: none;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e5e7eb;
        }}

        .conv-dropdown.open {{
            display: block;
        }}

        .chat-container {{
            background: #f9fafb;
            padding: 15px;
            border-radius: 8px;
            max-height: 500px;
            overflow-y: auto;
        }}

        .message-bubble {{
            margin-bottom: 12px;
            padding: 10px 14px;
            border-radius: 12px;
            max-width: 75%;
            word-wrap: break-word;
            animation: fadeIn 0.2s ease;
        }}

        .message-bubble.outbound {{
            background: #4f46e5;
            color: white;
            margin-left: auto;
            margin-right: 0;
            border-bottom-right-radius: 4px;
        }}

        .message-bubble.inbound {{
            background: #ffffff;
            color: #1f2937;
            margin-right: auto;
            margin-left: 0;
            border: 1px solid #e5e7eb;
            border-bottom-left-radius: 4px;
        }}

        .message-header {{
            font-size: 0.75em;
            font-weight: 600;
            margin-bottom: 4px;
            opacity: 0.8;
        }}

        .message-bubble.outbound .message-header {{
            color: rgba(255, 255, 255, 0.9);
        }}

        .message-bubble.inbound .message-header {{
            color: #6b7280;
        }}

        .message-content {{
            font-size: 0.9em;
            line-height: 1.5;
            white-space: pre-wrap;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(-5px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .r2-value {{
            color: #10b981;
            font-weight: 600;
            font-size: 0.9em;
        }}

        .r2-zero {{
            color: #9ca3af;
            font-size: 0.9em;
        }}

        .conv-id {{
            color: #4f46e5;
            font-weight: 600;
        }}

        .outcome-badge {{
            background: #10b981;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            margin-left: 6px;
        }}

        .raw-report-content {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.85em;
            line-height: 1.5;
            max-height: 600px;
            overflow-y: auto;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>{title}</h1>
            <p>Actionable insights and recommendations based on conversation analysis</p>
        </div>

        <!-- Summary Statistics -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(report.recommendations)}</div>
                <div class="stat-label">Total Recommendations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{report.total_conversations_analyzed}</div>
                <div class="stat-label">Conversations Analyzed</div>
            </div>
        </div>

        <!-- Analysis Summary -->
        <div class="section">
            <h2>Analysis Summary</h2>
            <div class="analysis-summary">{_escape_html(report.analysis_summary)}</div>
        </div>

        <!-- Recommendations -->
        <div class="section">
            <h2>Recommendations</h2>
'''

    # Add each recommendation
    for idx, rec in enumerate(report.recommendations, 1):
        html_content += f'''
            <div class="recommendation-card">
                <div class="rec-header">
                    <h3>{idx}. {_escape_html(rec.title)}</h3>
                </div>

                <div class="rec-section">
                    <h4>Rationale</h4>
                    <p>{_escape_html(rec.rationale)}</p>
                </div>

                <div class="rec-section">
                    <h4>Description</h4>
                    <p>{_escape_html(rec.description)}</p>
                </div>

                <div class="rec-section">
                    <h4>Evidence</h4>
                    <p>{_escape_html(rec.evidence)}</p>
                </div>
'''

        # Add supporting features if available
        if rec.supporting_features:
            html_content += f'''
                <div class="rec-section">
                    <h4>Supporting Features ({len(rec.supporting_features)})</h4>
                    <ul class="feature-list">
'''
            for feat in rec.supporting_features:
                r2_class = 'r2-value' if feat.r_squared > 0 else 'r2-zero'
                html_content += f'''
                        <li>
                            <strong>{_escape_html(feat.name)}</strong>
                            <span class="{r2_class}"> (R²: {feat.r_squared:.4f})</span>
                        </li>
'''
            html_content += '''
                    </ul>
                </div>
'''

        # Add supporting conversations if available
        if rec.supporting_conversations:
            html_content += f'''
                <div class="rec-section">
                    <h4>Supporting Conversations ({len(rec.supporting_conversations)})</h4>
                    <ul class="conversation-list">
'''
            for conv in rec.supporting_conversations:
                # Get conversation metadata
                conv_metadata = report.conversations[conv.conversation_id]
                conv_id_str = f'rec{idx}_conv{conv.conversation_id}'
                
                html_content += f'''
                        <li>
                            <div class="conv-header">
                                <div>
                                    <span class="conv-id">Conversation #{conv.conversation_id}</span>
                                    <span class="outcome-badge">{_escape_html(conv_metadata.outcome)}</span>
                                </div>
                                <button class="conv-toggle" onclick="toggleConversation('{conv_id_str}')">View Chat</button>
                            </div>
                            <div style="margin-top: 8px; color: #6b7280; font-size: 0.9em;">
                                {_escape_html(conv.explanation)}
                            </div>
                            <div class="conv-dropdown" id="{conv_id_str}">
                                <div class="chat-container">
'''
                
                # Add messages
                for msg in conv_metadata.conversation.messages:
                    # Determine message direction based on role
                    direction = 'outbound' if msg.role.lower() in ['agent', 'outbound', 'assistant'] else 'inbound'
                    timestamp_str = msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp)
                    
                    html_content += f'''
                                    <div class="message-bubble {direction}">
                                        <div class="message-header">[{timestamp_str}] [{msg.role}]</div>
                                        <div class="message-content">{_escape_html(msg.content)}</div>
                                    </div>
'''
                
                html_content += '''
                                </div>
                            </div>
                        </li>
'''
            html_content += '''
                    </ul>
                </div>
'''

        html_content += '''
            </div>
'''

    # Add raw report section
    html_content += '''
        </div>

        <!-- Features List -->
        <div class="section">
            <h2>Insightful Features Input to the Action Recommendation Analysis</h2>
            <details>
                <summary style="cursor: pointer; font-weight: 600; padding: 10px; background: #f9fafb; border-radius: 6px; margin-bottom: 10px;">
                    View all features analyzed (click to expand)
                </summary>
                <div style="margin-top: 15px;">
                    <p style="color: #6b7280; margin-bottom: 15px;">
                        These features were discovered during semantic analysis and ranked by their predictive power (R²).
                    </p>
                    <ol style="line-height: 2; color: #374151;">
'''
    
    # Add each feature
    for feature in features_list:
        html_content += f'''
                        <li>{_escape_html(feature)}</li>
'''
    
    html_content += '''
                    </ol>
                </div>
            </details>
        </div>

        <!-- Full Report -->
        <div class="section">
            <h2>Full Report (text version)</h2>
            <div class="raw-report-content">'''
    
    html_content += _escape_html(report.raw_report)
    
    html_content += '''</div>
        </div>
    </div>

    <script>
        function toggleConversation(id) {
            const dropdown = document.getElementById(id);
            dropdown.classList.toggle('open');
        }
    </script>
</body>
</html>
'''

    # Write to file
    output_file.write_text(html_content, encoding='utf-8')
    
    return output_file
