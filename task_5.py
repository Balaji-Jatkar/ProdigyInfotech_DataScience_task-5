import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
from io import StringIO

print("=== TRAFFIC ACCIDENT DATA ANALYSIS ===")
print("Loading real traffic accident data from FARS database...")

url = "https://raw.githubusercontent.com/DS202-at-ISU/labs/master/data/fars2016/accident.csv"

try:
    df = pd.read_csv(url)
    print(f"✓ Successfully loaded FARS 2016 accident data")
    print(f"Dataset shape: {df.shape[0]:,} accidents, {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print(f"\nColumns available: {list(df.columns)}")

print("\n=== 1. TIME PATTERNS ===")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
hourly = df['HOUR'].value_counts().sort_index()
hourly = hourly[hourly.index < 24]
plt.bar(hourly.index, hourly.values, color='skyblue', alpha=0.8)
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour (24h format)')
plt.ylabel('Number of Accidents')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 3, 2)
day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
weekly = df['DAY_WEEK'].value_counts().sort_index()
plt.bar(range(len(weekly)), weekly.values, color='lightcoral', alpha=0.8)
plt.title('Accidents by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Accidents')
plt.xticks(range(len(day_names)), day_names, rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 3, 3)
monthly = df['MONTH'].value_counts().sort_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.bar(monthly.index, monthly.values, color='lightgreen', alpha=0.8)
plt.title('Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.xticks(monthly.index, [month_names[i-1] for i in monthly.index])
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"🕐 Peak accident hour: {hourly.idxmax()}:00 ({hourly.max():,} accidents)")
print(f"📅 Peak accident day: {day_names[weekly.idxmax()-1]} ({weekly.max():,} accidents)")
print(f"📆 Peak accident month: {month_names[monthly.idxmax()-1]} ({monthly.max():,} accidents)")

print("\n=== 2. WEATHER CONDITIONS ===")

weather_map = {
    1: 'Clear', 2: 'Rain', 3: 'Sleet/Hail', 4: 'Snow', 
    5: 'Fog/Smog/Smoke', 6: 'Severe Crosswinds', 7: 'Blowing Sand/Dirt',
    8: 'Other', 10: 'Cloudy', 11: 'Blowing Snow', 98: 'Not Reported', 99: 'Unknown'
}

atmos_map = {
    1: 'Clear', 2: 'Rain', 3: 'Sleet/Hail', 4: 'Snow',
    5: 'Fog/Smog/Smoke', 6: 'Severe Crosswinds', 7: 'Blowing Sand/Dirt',
    8: 'Other', 10: 'Cloudy', 11: 'Blowing Snow', 98: 'Not Reported', 99: 'Unknown'
}

plt.figure(figsize=(15, 10))

if 'WEATHER' in df.columns:
    weather = df['WEATHER'].map(weather_map).value_counts().head(8)
    
    plt.subplot(2, 2, 1)
    plt.bar(weather.index, weather.values, color='steelblue', alpha=0.8)
    plt.title('Accidents by Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Number of Accidents')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

light_map = {
    1: 'Daylight', 2: 'Dark - Street Lights', 3: 'Dark - No Street Lights',
    4: 'Dawn', 5: 'Dusk', 6: 'Dark - Unknown Lighting',
    7: 'Other', 8: 'Not Reported', 9: 'Unknown'
}

if 'LGT_COND' in df.columns:
    light = df['LGT_COND'].map(light_map).value_counts().head(6)
    
    plt.subplot(2, 2, 2)
    colors = plt.cm.Set3(np.linspace(0, 1, len(light)))
    wedges, texts, autotexts = plt.pie(light.values, autopct='%1.1f%%', startangle=90, colors=colors, 
                                      textprops={'fontsize': 9})
    plt.title('Light Conditions During Accidents', fontsize=12, pad=20)
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.legend(wedges, light.index, title="Light Conditions", loc="center left", 
               bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)

if 'ATMOSPH_COND' in df.columns and 'ATMOSPH_COND' != 'WEATHER':
    atmos = df['ATMOSPH_COND'].map(atmos_map).value_counts().head(6)
    
    plt.subplot(2, 2, 3)
    plt.barh(atmos.index, atmos.values, color='orange', alpha=0.8)
    plt.title('Atmospheric Conditions')
    plt.xlabel('Number of Accidents')
    plt.grid(axis='x', alpha=0.3)

if 'WEATHER' in df.columns and 'HOUR' in df.columns:
    weather_hour = pd.crosstab(df['HOUR'], df['WEATHER'].map(weather_map))
    weather_hour = weather_hour[(weather_hour.index < 24)]
    main_weather = ['Clear', 'Rain', 'Snow', 'Fog/Smog/Smoke', 'Cloudy']
    available_weather = [w for w in main_weather if w in weather_hour.columns]
    
    if available_weather:
        plt.subplot(2, 2, 4)
        for weather_type in available_weather[:4]:
            plt.plot(weather_hour.index, weather_hour[weather_type], 
                    marker='o', label=weather_type, alpha=0.8)
        plt.title('Weather Patterns by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Accidents')
        plt.legend()
        plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

if 'WEATHER' in df.columns:
    most_common_weather = df['WEATHER'].map(weather_map).value_counts().index[0]
    print(f"🌤️  Most common weather during accidents: {most_common_weather}")
    
    clear_accidents = len(df[df['WEATHER'] == 1])
    bad_weather_codes = [2, 3, 4, 5, 11]
    bad_weather_accidents = len(df[df['WEATHER'].isin(bad_weather_codes)])
    total_weather_reported = len(df[df['WEATHER'].notna() & (df['WEATHER'] < 90)])
    
    if total_weather_reported > 0:
        clear_pct = (clear_accidents / total_weather_reported) * 100
        bad_weather_pct = (bad_weather_accidents / total_weather_reported) * 100
        print(f"🌞 Clear weather accidents: {clear_pct:.1f}%")
        print(f"🌧️  Bad weather accidents: {bad_weather_pct:.1f}%")

if 'LGT_COND' in df.columns:
    daylight_accidents = (df['LGT_COND'] == 1).sum()
    dark_accidents = df['LGT_COND'].isin([2, 3, 6]).sum()
    total_light_reported = len(df[df['LGT_COND'].notna() & (df['LGT_COND'] < 8)])
    
    if total_light_reported > 0:
        daylight_pct = (daylight_accidents / total_light_reported) * 100
        dark_pct = (dark_accidents / total_light_reported) * 100
        print(f"☀️  Daylight accidents: {daylight_pct:.1f}%")
        print(f"🌙 Dark condition accidents: {dark_pct:.1f}%")

print("\n=== 3. ROAD CONDITIONS ===")

rural_urban_map = {1: 'Rural', 2: 'Urban'}
if 'RUR_URB' in df.columns:
    rural_urban = df['RUR_URB'].map(rural_urban_map).value_counts()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    colors = ['lightblue', 'orange']
    wedges, texts, autotexts = plt.pie(rural_urban.values, autopct='%1.1f%%', 
                                      colors=colors, startangle=90, textprops={'fontsize': 10})
    plt.title('Rural vs Urban Accidents', fontsize=12, pad=20)
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.legend(wedges, rural_urban.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

work_zone_map = {0: 'No Work Zone', 1: 'Work Zone'}
if 'WRK_ZONE' in df.columns:
    work_zone = df['WRK_ZONE'].map(work_zone_map).value_counts()
    
    plt.subplot(1, 2, 2)
    colors = ['lightcoral', 'yellow']
    wedges, texts, autotexts = plt.pie(work_zone.values, autopct='%1.1f%%',
                                      colors=colors, startangle=90, textprops={'fontsize': 10})
    plt.title('Work Zone Related Accidents', fontsize=12, pad=20)
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    plt.legend(wedges, work_zone.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.show()

print("\n=== 3. ACCIDENT SEVERITY ===")

if 'PERSONS' in df.columns:
    fatality_dist = df['PERSONS'].value_counts().head(10).sort_index()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(fatality_dist.index, fatality_dist.values, color='red', alpha=0.7)
    plt.title('Distribution of People Involved')
    plt.xlabel('Number of People')
    plt.ylabel('Number of Accidents')
    plt.grid(axis='y', alpha=0.3)

if 'VE_TOTAL' in df.columns:
    vehicle_dist = df['VE_TOTAL'].value_counts().head(8).sort_index()
    
    plt.subplot(1, 2, 2)
    plt.bar(vehicle_dist.index, vehicle_dist.values, color='purple', alpha=0.7)
    plt.title('Vehicles Involved in Accidents')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Number of Accidents')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\n=== 4. ACCIDENT HOTSPOTS ===")

state_codes = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
    8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'DC', 12: 'Florida',
    13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois', 18: 'Indiana',
    19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine',
    24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota',
    28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska',
    32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico',
    36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio',
    40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island',
    45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee', 48: 'Texas',
    49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington',
    54: 'West Virginia', 55: 'Wisconsin', 56: 'Wyoming'
}

state_accidents = df['STATE'].map(state_codes).value_counts().head(15)

plt.figure(figsize=(14, 8))
plt.barh(state_accidents.index, state_accidents.values, color='darkblue', alpha=0.7)
plt.title('Top 15 States with Most Fatal Accidents (2016)')
plt.xlabel('Number of Fatal Accidents')
plt.ylabel('State')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"🎯 Top accident state: {state_accidents.index[0]} ({state_accidents.iloc[0]:,} accidents)")

print("\n=== 5. ROADWAY ANALYSIS ===")

func_sys_map = {
    1: 'Interstate', 2: 'Other Freeways', 3: 'Other Principal Arterial',
    4: 'Minor Arterial', 5: 'Major Collector', 6: 'Minor Collector', 7: 'Local'
}

if 'FUNC_SYS' in df.columns:
    road_type = df['FUNC_SYS'].map(func_sys_map).value_counts()
    
    plt.figure(figsize=(12, 6))
    plt.barh(road_type.index, road_type.values, color='green', alpha=0.7)
    plt.title('Fatal Accidents by Road Type')
    plt.xlabel('Number of Accidents')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\n=== 📊 KEY INSIGHTS ===")
print(f"1. Total fatal accidents analyzed: {len(df):,}")
print(f"2. Peak accident hour: {hourly.idxmax()}:00 (late evening/night)")
print(f"3. Peak accident day: {day_names[weekly.idxmax()-1]}")
print(f"4. Deadliest month: {month_names[monthly.idxmax()-1]}")
print(f"5. Most common weather: {most_common_weather if 'WEATHER' in df.columns else 'N/A'}")
print(f"6. Highest accident state: {state_accidents.index[0]}")

if 'RUR_URB' in df.columns:
    rural_pct = (df['RUR_URB'] == 1).mean() * 100
    print(f"7. Rural vs Urban: {rural_pct:.1f}% rural, {100-rural_pct:.1f}% urban")

if 'WEATHER' in df.columns:
    bad_weather_rate = (df['WEATHER'].isin([2,3,4,5,11]).sum() / len(df[df['WEATHER'].notna()])) * 100
    print(f"8. Bad weather accidents: {bad_weather_rate:.1f}%")

if 'LGT_COND' in df.columns:
    dark_rate = (df['LGT_COND'].isin([2,3,6]).sum() / len(df[df['LGT_COND'].notna()])) * 100
    print(f"9. Dark condition accidents: {dark_rate:.1f}%")

if 'VE_TOTAL' in df.columns:
    single_vehicle = (df['VE_TOTAL'] == 1).mean() * 100
    print(f"10. Single vehicle accidents: {single_vehicle:.1f}%")

avg_people = df['PERSONS'].mean()
print(f"11. Average people per accident: {avg_people:.1f}")

print("\n🎯 SAFETY RECOMMENDATIONS:")
print("• Increase enforcement during evening hours (6-9 PM)")
print("• Focus weekend safety campaigns")
print("• Weather-specific alerts during rain/snow/fog conditions")
print("• Improve lighting on dark roads without street lights")
print("• Enhanced rural road safety measures")
print("• State-specific targeted interventions")

print(f"\n📍 ACCIDENT HOTSPOT MAP:")
if 'LATITUDE' in df.columns and 'LONGITUD' in df.columns:
    sample_df = df.sample(n=min(1000, len(df)))
    valid_coords = sample_df[(sample_df['LATITUDE'] > 0) & (sample_df['LONGITUD'] < 0)]
    print(f"Valid coordinates available for {len(valid_coords)} accidents")
    print("Use LATITUDE and LONGITUD columns for mapping hotspots")
else:
    print("Geographic coordinates not available in this dataset")

print("\n=== ANALYSIS COMPLETE ===")
print("Data source: FARS (Fatality Analysis Reporting System) 2016")