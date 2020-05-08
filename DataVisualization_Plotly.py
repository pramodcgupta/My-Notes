###################  1. Categorical Plots
# ------------------------------------------------- Pie Plot ---------------------------------------------------------------
# Categorical columns plot pie
#function  for pie plot for customer attrition types

# Import library for plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

# Define Class1 and Class2 Dataframe
df_no=df[df['y'] == 'no']
df_yes=df[df['y'] != 'no']

# Separte categorical and Numerical variables
cat_cols=df.select_dtypes(exclude=['float_','number','bool_'])
num_cols=df.select_dtypes(exclude=['object','bool_'])
# Define target 
target_col=['y']
# Exclude target from categorical variable
cat_cols=[cat for cat in cat_cols if cat not in target_col]


def plot_pie(column, class1_dataFrame, class2_dataFrame, class1_name, class2_name) :
    
    
    trace1 = go.Pie(values  = class1_dataFrame[column].value_counts().values.tolist(),
                    labels  = class1_dataFrame[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = class1_name,
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6
                   )
    trace2 = go.Pie(values  = class2_dataFrame[column].value_counts().values.tolist(),
                    labels  = class2_dataFrame[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = class2_name
                   )


    layout = go.Layout(dict(title = column + " distribution in Bank Subscription ",  # Update title
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = class1_name,
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .15, y = .5),
                                           dict(text = class2_name,
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .88,y = .5
                                               )
                                          ]
                           )
                      )
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    py.iplot(fig)
    
#for all categorical columns plot pie

class1_dataFrame=df_no   # Put Class 1 dataframe
class2_dataFrame=df_yes  # Put Class 2 dataframe
class1_name='No Subscription'  # Put Class 1 Name
class2_name='Yes Subscription'  # Put Class 2 Name

for i in cat_cols:
    plot_pie(i,class1_dataFrame, class2_dataFrame, class1_name, class2_name) 

# ------------------------------------------------- Bar Graph ---------------------------------------------------------------
# Categorical columns Bar pie
#cusomer attrition in tenure groups
tg_ch  =  churn["tenure_group"].value_counts().reset_index()
tg_ch.columns  = ["tenure_group","count"]
tg_nch =  not_churn["tenure_group"].value_counts().reset_index()
tg_nch.columns = ["tenure_group","count"]

#bar - churn
trace1 = go.Bar(x = tg_ch["tenure_group"]  , y = tg_ch["count"],
                name = "Churn Customers",
                marker = dict(line = dict(width = .5,color = "black")),
                opacity = .9)

#bar - not churn
trace2 = go.Bar(x = tg_nch["tenure_group"] , y = tg_nch["count"],
                name = "Non Churn Customers",
                marker = dict(line = dict(width = .5,color = "black")),
                opacity = .9)

layout = go.Layout(dict(title = "Customer attrition in tenure groups",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "tenure group",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "count",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                       )
                  )
data = [trace1,trace2]
fig  = go.Figure(data=data,layout=layout)
py.iplot(fig)



###################  2. Numerical Plots
# ------------------------------------------------- Histogram ---------------------------------------------------------------
# Numerical columns plot : histogram  
#function  for histogram for customer attrition types
def histogram(column) :
    trace1 = go.Histogram(x  = churn[column],
                          histnorm= "percent",
                          name = "Churn Customers",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    
    trace2 = go.Histogram(x  = not_churn[column],
                          histnorm = "percent",
                          name = "Non churn customers",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in customer attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)
    
#for all Numerical columns plot histogram    
for i in num_cols :
    histogram(i)    
    
# --------------------------------------------------- Seaborn Plot ------------------------------------------------------------------------
# Distribution of monthly charges by churn    
    
ax = sns.kdeplot(telecom_cust.MonthlyCharges[(telecom_cust["Churn"] == 'No')], color="Red", shade = True)
ax = sns.kdeplot(telecom_cust.MonthlyCharges[(telecom_cust["Churn"] == 'Yes')], ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of monthly charges by churn')    
    