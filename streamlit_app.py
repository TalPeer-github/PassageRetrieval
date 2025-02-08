import streamlit as st
import pandas as pd
import altair as alt
import ast 
import matplotlib.pyplot as plt
import seaborn as sns 

st.set_page_config(
    page_title="Passage Retrieval - Sleek Home Assignment",
    page_icon="🏂",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
st.title("Sleek Home Assignment - Tal Peer")
st.header("Explore our results!")
good_positions = ['Data Scientist', 'Data Engineer', 'Accountant', 'Software Architect','Java Developer','Facilities Project Manager','Licensed Therapist','Division Manager']
good_persons =
position_summaries = {
    'Data Scientist': {
        'education_summary': "Data Scientists typically hold advanced degrees in fields such as computer science, mathematics, statistics, or related disciplines.",
        'skills_summary': "Key skills for Data Scientists include proficiency in programming languages like Python or R, data manipulation and analysis, machine learning, and data visualization.",
        'experience_summary': "Data Scientists often have experience working with large datasets, developing predictive models, and deploying machine learning solutions in real-world scenarios."
    },
    'Data Engineer': {
        'education_summary': "Data Engineers commonly have degrees in computer science, software engineering, or related fields.",
        'skills_summary': "Data Engineers possess skills in data modeling, database management, ETL (Extract, Transform, Load) processes, and proficiency in programming languages like Python, SQL, or Java.",
        'experience_summary': "Data Engineers typically have experience designing and implementing data pipelines, optimizing database performance, and working with distributed computing frameworks like Hadoop or Spark."
    },
    'Accountant': {
        'education_summary': "Accountants often hold bachelor's degrees in accounting, finance, or related fields, and may pursue certifications such as CPA (Certified Public Accountant) or CMA (Certified Management Accountant).",
        'skills_summary': "Accountants possess skills in financial analysis, bookkeeping, tax preparation, audit procedures, and proficiency in accounting software.",
        'experience_summary': "Accountants gain experience in preparing financial statements, conducting audits, analyzing financial data, and ensuring compliance with regulations and accounting standards."
    },
    'Software Architect': {
        'education_summary': "Software Architects typically hold bachelor's or master's degrees in computer science, software engineering, or related fields.",
        'skills_summary': "Software Architects possess skills in system design, software development methodologies, architectural patterns, and proficiency in programming languages like Java, C#, or Python.",
        'experience_summary': "Software Architects have experience leading software development teams, designing scalable and reliable systems, and making architectural decisions to meet business requirements."
    },
    'Java Developer': {
        'education_summary': "Java Developers commonly have bachelor's degrees in computer science, software engineering, or related fields.",
        'skills_summary': "Java Developers possess skills in Java programming, object-oriented design, web development frameworks like Spring or Hibernate, and proficiency in software development tools and technologies.",
        'experience_summary': "Java Developers gain experience in developing Java-based applications, implementing software solutions, debugging and troubleshooting code, and collaborating with cross-functional teams."
    },
    'Facilities Project Manager': {
        'education_summary': "Facilities Project Managers often have bachelor's degrees in engineering, construction management, or related fields, and may have certifications such as PMP (Project Management Professional).",
        'skills_summary': "Facilities Project Managers possess skills in project management, construction planning, budgeting, scheduling, and coordination of facility-related projects.",
        'experience_summary': "Facilities Project Managers have experience overseeing construction projects, managing contractors and vendors, ensuring compliance with regulations, and delivering projects on time and within budget."
    },
    'Licensed Therapist': {
        'education_summary': "Licensed Therapists typically hold master's or doctoral degrees in psychology, counseling, social work, or related fields, and are licensed to practice therapy.",
        'skills_summary': "Licensed Therapists possess skills in psychotherapy techniques, assessment and diagnosis, treatment planning, and therapeutic communication.",
        'experience_summary': "Licensed Therapists gain experience working with diverse populations, conducting individual or group therapy sessions, addressing mental health concerns, and collaborating with other healthcare professionals."
    },
    'Division Manager': {
        'education_summary': "Division Managers often have bachelor's or master's degrees in business administration, management, or related fields.",
        'skills_summary': "Division Managers possess skills in leadership, strategic planning, decision-making, team management, and effective communication.",
        'experience_summary': "Division Managers have experience overseeing operations, managing teams, setting goals and objectives, and driving performance to achieve organizational success."
    }
}

df_reshaped = pd.read_csv('data/final_EM.csv')
df_reshaped = df_reshaped[df_reshaped['position'].isin(good_positions)]
df_reshaped['education'] =df_reshaped['education'].apply(ast.literal_eval)
df_reshaped['skills'] =df_reshaped['skills'].apply(ast.literal_eval)
df_reshaped['years_of_experience'] =df_reshaped['years_of_experience'].apply(ast.literal_eval)


positions_list = list(df_reshaped['position'].unique())[::-1]

st.header('Position Selection')
st.write('Please select a position from the list below:')
selected_position = st.selectbox('Select Position', positions_list)

st.subheader(f'You selected: {selected_position}\n')
colors = ['#164863', '#427D9D', '#9BBEC8', '#DDF2FD', '#F2EBE9', '#6B818C']

def plot_education(position_title):
    position_data = df_reshaped[df_reshaped['position'] == position_title]
    education_counts = position_data['education'].iloc[0]
    sorted_education_counts = dict(sorted(education_counts.items(), key=lambda x: x[1], reverse=True))
    top_5_education = dict(list(sorted_education_counts.items())[:5])
    all_other_count = sum(sorted_education_counts.values()) - sum(top_5_education.values())
    combined_education_counts = {**top_5_education, 'All Other': all_other_count}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(combined_education_counts.values(), labels=combined_education_counts.keys(), autopct='%1.2f%%',
           startangle=-140, colors=colors,textprops={'fontsize': 10})
    ax.set_title(f"Educational Levels for {position_title}\n\n")
    ax.axis('equal')
    st.pyplot(fig)

def plot_skills(position_title):
    position_data = df_reshaped[df_reshaped['position'] == position_title]
    skills_counts = position_data['skills'].iloc[0]
    total_skills = sum(skills_counts.values())
    skills_percentages = {skill: (count / total_skills) * 100 for skill, count in
                          position_data['skills'].iloc[0].items()}
    skills_df = pd.DataFrame(list(skills_percentages.items()), columns=['skill', 'p'])
    skills_df = skills_df.sort_values(by='p', ascending=True)

    num_skills_to_display = min(15, len(skills_df))
    skills_df = skills_df.iloc[-num_skills_to_display:]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    plt.barh(skills_df['skill'], skills_df['p'], color='#DDF2FD', edgecolor='#0A065D', alpha=0.8)
    plt.xlabel('Percentage (%)')
    plt.ylabel('Skill')
    plt.title(f'Top Skills for {position_title} (%)\n\n')

    st.pyplot(fig)

def plot_experience(position_title):
    position_data = df_reshaped[df_reshaped['position'] == position_title]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    years_of_experience = position_data['years_of_experience'].explode().reset_index()
    sns.kdeplot(data=years_of_experience,x='years_of_experience', ax=axes[0],palette=['#9BBEC8'],shade=True,linewidth=2,alpha=.3)
    axes[0].set_title(f"Years of Experience for {position_title}")
    axes[0].set_xlabel("Years of Experience")
    axes[0].set_ylabel("Frequency")

    sns.boxplot(years_of_experience['years_of_experience'],ax=axes[1], vert=True,palette=['#9BBEC8'],linewidth=1.5,saturation=1,medianprops={"color": "#164863", "linewidth": 1 ,"linestyle":'--'})
    axes[1].set_title(f"Years of Experience for {position_title}")
    axes[1].set_xlabel("Years of Experience")
    plt.tight_layout()
    
    st.pyplot(fig)

container1 = st.container(border=True)
st.write(position_summaries[selected_position]['education_summary'])
with container1:
    plot_education(selected_position)
container2 = st.container(border=True)
st.write(position_summaries[selected_position]['skills_summary'])
with container2:
    plot_skills(selected_position)
container3 = st.container(border=True)
st.write(position_summaries[selected_position]['experience_summary'])
with container3:
    plot_experience(selected_position)
