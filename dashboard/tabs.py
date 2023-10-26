from  explainerdashboard.custom import *
import dash
from dash import html
import dash_bootstrap_components as dbc
from explainerdashboard import ClassifierExplainer, ExplainerDashboard


class ModelSummaryTab(ExplainerComponent):
    """A class for generating the Model Performance Overview Tab within the dashboard."""

    def __init__(self, explainer, name=None, **kwargs):
        super().__init__(explainer, name=name, title='Model Performance Overview', **kwargs)

        self.roc_auc = RocAucComponent(explainer,
                                       hide_subtitle=True, hide_footer=True,
                                       hide_popout=True, **kwargs)
        
        self.confusion_matrix = ConfusionMatrixComponent(explainer,
                                                         hide_subtitle=True, hide_footer=True,
                                                         hide_binary=True, hide_selector=True,
                                                         hide_popout=True, **kwargs)

        self.shap_summary = ShapSummaryComponent(explainer,
                                                 hide_depth=True, depth=20,
                                                 hide_type=True, hide_selector=True,
                                                 hide_subtitle=True, **kwargs)

    def layout(self):

        return dbc.Container([
            html.Div(html.H1('Model performance'), style={'margin': '25px 0 35px 5px'}),

             html.Div([
                html.Div(html.P('''
                                The Random Forest model has shown promising performance in the classification task,
                                achieving a ROC AUC score of 93%.
                                This score provides an indication of the model's ability to predict
                                whether online store users will make a purchase at the end of their current session
                                when using the default threshold of 0.5.
                                '''), style={'width': '50%', 'margin': '5px', 'fontSize':'18px'}),
                html.Div(html.P('''
                                It's essential to acknowledge the presence of a significant class imbalance in the dataset.
                                Under these circumstances, the Random Forest model,
                                trained on the complete feature set and with fine-tuned hyperparameters,
                                has demonstrated the capacity to strike a balance between false positives and false negatives,
                                as observed from the confusion matrix.
                                '''), style={'width': '50%', 'margin': '5px', 'fontSize':'18px'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'height': '70%'}),

            html.Div([
                html.Div(self.roc_auc.layout(), style={'width': '50%', 'margin': '5px'}),
                html.Div(self.confusion_matrix.layout(), style={'width': '50%', 'margin': '5px'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'height': '70%'}),

            html.Div(html.P('''
                            In the top 20 features that have the most significant influence
                            on the model's predictions, all numerical features
                            and some categorical features are included.
                            The feature with the most substantial impact on the model's decision
                            is PageValues, which represents the average value of pages visited by the user.
                            On average, this feature alters the predicted probability of a user making a purchase by more than 20%. 
                            '''), style={'margin': '15px 0 0 0', 'fontSize': '18px'}),
            
            html.Div(self.shap_summary.layout(), style={'margin': '15px 0 0 0'})
        ])

class PredictionsTab(ExplainerComponent):
    """A class for generating the Individual Predictions Tab within the dashboard."""

    def __init__(self, explainer, name=None, **kwargs):
        super().__init__(explainer, name=name, title='Individual Predictions', **kwargs)

        self.index = ClassifierRandomIndexComponent(explainer, index=0,
                                                    hide_title=True, hide_subtitle=True,
                                                    hide_slider=True, hide_labels=True,
                                                    hide_pred_or_perc=True, hide_index=False,
                                                    hide_selector=True, hide_button=False,
                                                    **kwargs)
        
        self.contributions = ShapContributionsGraphComponent(explainer,  index=0,
                                                             hide_title=True,hide_index=True,
                                                             hide_subtitle=True,
                                                             hide_depth=True, depth=8,
                                                             hide_sort=True, hide_orientation=True,
                                                             hide_cats=True, hide_selector=True,
                                                             sort='importance', **kwargs)
        
        self.pred_summary = ClassifierPredictionSummaryComponent(explainer, hide_title=True, index=0,
                                                                 hide_index=True, hide_subtitle=True,
                                                                 index_dropdown=True, pos_label=None,
                                                                 hide_selector=True, **kwargs)
        
        self.connector = IndexConnector(self.index, [self.contributions, self.pred_summary])


    def layout(self):
        return dbc.Container([
            html.Div(html.H1('Individual predictions'), style={'margin': '25px'}),

            dbc.Row([
                html.Div([
                html.Div(html.H4('Choose customer index:'), style={'margin': '25px'}),
                html.Div(self.index.layout(), style={'width': '40%'})
            ], style={'display': 'flex', 'align-items': 'center'}), 
        ]),

            dbc.Row([
                html.H3('Prediction summary'),
                self.pred_summary.layout()
            ], align='center', justify='between', style={'margin': '25px', 'background-color': 'lightgray', 'border-radius': '5px'}),
            
            dbc.Row([
                html.H3('Contributions to prediction'),
                self.contributions.layout(),
            ], align='center', justify='between', style={'margin': '25px', 'background-color': 'lightgray', 'border-radius': '5px'}),
        ])
    