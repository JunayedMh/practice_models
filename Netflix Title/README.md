# Netflix Content Analysis 🎬

This project was not about building a prediction model. It was more about exploring a dataset and asking questions like where does Netflix get most of its content from? When did Netflix really start growing? Is it mostly movies or TV shows? This kind of work is called **EDA** and it's something you do before any machine learning to actually understand what you're working with.

## What dataset did I use?

The dataset was `netflix_titles.csv` which had **8,807 titles** and **12 columns**. Each row was one piece of content either a movie or a TV show. The columns were things like title, director, cast, country, date added to Netflix, release year, rating, duration, genre, and description.

## What did I find about missing data?

One of the first things I checked was which columns had missing values. This is always important because missing data can mess up your analysis if you don't know about it. Here's what I found:

| Column | Missing Values |
|---|---|
| `director` | 2,634 missing |
| `cast` | 825 missing |
| `country` | 831 missing |
| `date_added` | 10 missing |
| `rating` | 4 missing |
| `duration` | 3 missing |

The `director` column had the most missing data by far 2,634 entries. This makes sense because a lot of content on Netflix doesn't have a credited director
.

## Movies vs TV Shows — which one dominates?

Netflix has way more movies than TV shows. Out of 8,807 titles:

- **Movies: 6,131**
- **TV Shows: 2,676**

So roughly 70% of Netflix's library is movies. That was a bit surprising honestly — I expected it to be more balanced.

## Which countries produce the most Netflix content?

The top 10 countries by content count were:

1. United States - 2,818
2. India - 972
3. United Kingdom - 419
4. Japan - 245
5. South Korea - 199
6. Canada - 181
7. Spain - 145
8. France - 124
9. Mexico - 110
10. Egypt - 106

The US dominates by a massive margin. India is a distant second but still very strong. South Korea showing up at number 5 makes sense given how popular K-dramas have become globally.

## When did Netflix start adding a lot of content?

I parsed the `date_added` column into year and month so I could see the growth over time. Here's the content added per year:

| Year | Titles Added |
|---|---|
| 2008 | 2 |
| 2009 | 2 |
| 2015 | 82 |
| 2016 | 429 |
| 2017 | 1,188 |
| 2018 | 1,649 |
| 2019 | 2,016 |
| 2020 | 1,879 |
| 2021 | 1,498 |

Netflix was basically doing nothing until 2016, then suddenly exploded. 2019 was the peak year with over 2,000 titles added. The slight drop in 2020 and 2021 is probably due to COVID slowing down production.

## What visualizations did I make?

I built a **choropleth world map** using Plotly that showed Netflix content count by country using a red color scale darker red meant more content. The US lit up the most, with India and the UK also visible. Most of Africa and Central Asia were nearly empty which tells you Netflix's content library is still heavily western-dominated.

## What did I learn from this project?

- How to check for missing values and understand what they mean
- How to parse date columns and extract year and month from them
- How to use `value_counts()` to quickly summarize categorical data
- How to split comma-separated values in a column (some titles had multiple countries listed) and count them properly
- How to build an interactive choropleth map with Plotly — this was new and pretty cool
- That EDA is about asking good questions first, then letting the data answer them

## What could be done next?

- Analyze which **genres** appear the most
- See which **months** Netflix adds the most content (bet it's October/November before the holiday season)
- Build a simple **recommender** based on genre and country
- Explore whether **ratings** (TV-MA, PG, etc.) have changed over the years
- Compare average movie duration vs TV show seasons

## Tech stack

- Python 3.12
- pandas
- matplotlib, seaborn
- Plotly Express (for the choropleth map)
- Dataset: Netflix titles dataset (8,807 titles)