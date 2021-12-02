#Find the avg of all numeric columns
from pyspark.sql.functions import when, lit, col , avg
from pyspark.sql.types import  StringType, IntegerType 

#sample a row from sparkdf and convert to dictionary for easy analysis 
def row2dic(sparkdf):
    return sparkdf.take(1)[0].asDict()
    

#return mean of each numeric column in spark df for missing value imputation.
def mean_of_pyspark_columns(df, numeric_cols, verbose=False, fill_missing= False):

    #calculate the mean value per columns 
    col_with_mean=[]
    for col in numeric_cols:
        mean_value = df.select(avg(df[col]))
        avg_col = mean_value.columns[0]
        res = mean_value.rdd.map(lambda row : row[avg_col]).collect()
        
        if (verbose==True): print(mean_value.columns[0], "\t", res[0])
        col_with_mean.append([col, res[0]])    
    
    #Fill missing values for mean for continous variables(using inner func becuase this will only be used when the mean of columns is calculated )
    def fill_missing_with_mean(df, numeric_cols):
        
        col_with_mean = mean_of_pyspark_columns(df, numeric_cols) 
        
        for col, mean in col_with_mean:
            df = df.withColumn(col, when(df[col].isNull()==True, 
            lit(mean)).otherwise(df[col]))

        return df

    #if you want to return a df with all cols with mean  imputed for missing data  
    if fill_missing: 
        return fill_missing_with_mean(df, numeric_cols)

    #if you want just the column means
    else:
        return col_with_mean

#calculate mode for categorical fields  
def mode_of_pyspark_columns(df, cat_col_list, verbose=False,fill_missing=False):
    col_with_mode=[]
    for col in cat_col_list:
        #Filter null
        df = df.filter(df[col].isNull()==False)
        #Find unique_values_with_count
        unique_classes = df.select(col).distinct().rdd.map(lambda x: x[0]).collect()
        unique_values_with_count=[]
        for uc in unique_classes:
             unique_values_with_count.append([uc, df.filter(df[col]==uc).count()])
        #sort unique values w.r.t their count values
        sorted_unique_values_with_count= sorted(unique_values_with_count, key = lambda x: x[1], reverse =True)
        
        if (verbose==True): print(col, sorted_unique_values_with_count, " and mode is ", sorted_unique_values_with_count[0][0])
        col_with_mode.append([col, sorted_unique_values_with_count[0][0]])
    
    
    def fill_missing_with_mode(df, cat_col_list):
        col_with_mode =mode_of_pyspark_columns(df, cat_col_list)
        
        for col, mode in col_with_mode:
            df = df.withColumn(col, when(df[col].isNull()==True, 
            lit(mode)).otherwise(df[col]))
            
        return df
    if fill_missing:
        return fill_missing_with_mode(df,cat_col_list)
    else:
        return col_with_mode




#return the unique values of a column in spark dataframe 
def pyspark_unique_col_vals(df, col):
    return df.select(col).distinct().rdd.map(lambda r: r[0].collect())

#return total missing values for a pyspark df column 
def pyspark_missing_col_count(df, col): 
    return df.filter(df[col].isNull()).count()


#automatic assignment of pyspark column dtypes to new schema 
def assign_sparkdf_coldtypes(spark_df):
    out = {}
    for k,v in spark_df.take(1)[0].asDict().items():
        try:
            float(v)
            out[k] = 'int'
        except:
            out[k] = 'string'

    for k,v in out.items():
        if v=='string':
            spark_df = spark_df.withColumn(k, col(k).cast(StringType()))
        else:
            spark_df = spark_df.withColumn(k, col(k).cast(IntegerType()))
    return spark_df