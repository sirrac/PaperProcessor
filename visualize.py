import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

class PaperVisualizer:
    def __init__(self, theme='plotly_white'):
        self.theme = theme
        self.colors = px.colors.qualitative.Set3
        
    def create_topic_network(self, papers_data):
        """Create interactive topic network visualization"""
        G = nx.Graph() #undirected
        
        # Create network from paper topics

        # construct a graph
        for paper in papers_data:
            topics = paper.get('topics', [])
            for i, topic1 in enumerate(topics):
                for topic2 in topics[i+1:]:
                    if G.has_edge(topic1, topic2):
                        G[topic1][topic2]['weight'] += 1
                    else:
                        G.add_edge(topic1, topic2, weight=1)
        
        # Convert to plotly figure

        #extract x and y and put them into lists so i can make a spring layout 
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(
                size=10,
                color=self.colors[:len(G.nodes())],
                line_width=2)))
        
        return fig
    
    def create_citation_heatmap(self, papers_data):
        """Create citation relationship heatmap"""
        # Create citation matrix
        papers = [p['title'] for p in papers_data]
        matrix = [[0 for _ in papers] for _ in papers]
        
        for i, paper in enumerate(papers_data):
            for citation in paper.get('citations', []):
                if citation in papers:
                    j = papers.index(citation)
                    matrix[i][j] += 1
        
        fig = px.imshow(matrix,
                       labels=dict(x="Cited Papers", y="Citing Papers"),
                       x=papers,
                       y=papers,
                       color_continuous_scale="Viridis")
        
        return fig
    
    def create_topic_trends(self, papers_data):
        """Visualize topic trends over time"""
        # Extract topics and dates
        topic_dates = []
        for paper in papers_data:
            date = paper.get('date')
            for topic in paper.get('topics', []):
                topic_dates.append({'topic': topic, 'date': date})
        
        df = pd.DataFrame(topic_dates)
        df['count'] = 1
        df_grouped = df.groupby(['date', 'topic'])['count'].sum().reset_index()
        
        fig = px.line(df_grouped, 
                     x='date', 
                     y='count', 
                     color='topic',
                     title='Topic Trends Over Time')
        
        return fig
    
    def create_wordcloud(self, text_data):
        """Generate wordcloud from paper content"""
        wordcloud = WordCloud(width=800, 
                            height=400,
                            background_color='white').generate(text_data)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        return plt
    
    def create_methodology_comparison(self, papers_data):
        """Compare different methodologies used across papers"""
        methods = {}
        for paper in papers_data:
            for method in paper.get('methodology', []):
                methods[method] = methods.get(method, 0) + 1
        
        fig = px.bar(x=list(methods.keys()),
                    y=list(methods.values()),
                    title='Methodology Comparison',
                    labels={'x': 'Methodology', 'y': 'Count'})
        
        return fig
    
    def create_dashboard(self, papers_data):
        """Create a comprehensive dashboard"""
        from dash import Dash, html, dcc
        import dash_bootstrap_components as dbc
        
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Paper Analysis Dashboard"), width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=self.create_topic_network(papers_data))
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=self.create_citation_heatmap(papers_data))
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=self.create_topic_trends(papers_data))
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=self.create_methodology_comparison(papers_data))
                ], width=12)
            ])
        ])
        
        return app

# Example usage
if __name__ == "__main__":
    visualizer = PaperVisualizer()
    
    # Sample data
    papers_data = [
        {
            'title': 'Paper 1',
            'topics': ['ML', 'NLP'],
            'methodology': ['Deep Learning', 'Transformers'],
            'date': '2023-01'
        },
        # Add more paper data...
    ]
    
    # Create individual visualizations
    topic_network = visualizer.create_topic_network(papers_data)
    citation_heatmap = visualizer.create_citation_heatmap(papers_data)
    
    # Create dashboard
    app = visualizer.create_dashboard(papers_data)
    app.run_server(debug=True)