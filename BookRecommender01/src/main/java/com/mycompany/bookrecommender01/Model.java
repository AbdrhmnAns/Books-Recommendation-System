/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.bookrecommender01;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.DataFrameWriter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author AbdrhmnAns
 */
public class Model {

    public static void main(String[] args) {
        SparkSession session = SparkSession.builder()
                .appName("BookRecommender")
                .master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///D://")
                .getOrCreate();

        Dataset<Row> dataset = session
                .read().option("header", "true")
                .csv("ratings.csv");
        dataset.show();
        
     /*     Dataset<Row> dataset2 = session
                .read().option("header", "true")
                .csv("output.csv");*/
        //dataset.show();
        //data.write().json("D:\\jsondata");
        ////Dataset<Row> mydata = session.read().json("D:\\jsondata\\m.json");
        ////mydata.show();
        //preprocessing
     
        JavaRDD<Rating> javaRDD = dataset.toJavaRDD().map(new Function<Row, Rating>() {
            @Override
            public Rating call(Row row) throws Exception {
                //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
                //   System.out.println("here " + row.toString());
                String[] fields = row.toString().split(",");
                //System.out.println("userid "+fields[0]);
                // System.err.println("ISBN "+fields[1]);

                // System.out.println("userid " + fields[2]);
                String userId = fields[0];
                int nuserId = Integer.parseInt(userId.replaceAll("[\\[\\]]",""));
                String ISBN = fields[1];
                int nISBN = Integer.parseInt(ISBN.replaceAll("[\\[\\]]",""));
                String rate = fields[2];
                int Ratee = Integer.parseInt(rate.replaceAll("[\\[\\]]",""));
                Rating r = new Rating();
                r.setUserId(nuserId);
                r.setRate(Ratee);
                r.setISBN(nISBN);
                return r;
            }
        });
        
        Dataset<Row> ratingDataset = session.createDataFrame(javaRDD, Rating.class);
        //  ratingDataset.write().json("E://cvbb");
        ratingDataset.show();
//dframe = dframe.withColumn("c_number", dframe.col("c_a").cast("decimal(38,0)"));
      /*  StringIndexerModel userIdLI = new StringIndexer().setInputCol("userId").fit(ratingDataset).setOutputCol("ouserId");
        StringIndexerModel ISBNLI = new StringIndexer().setInputCol("ISBN").fit(ratingDataset).setOutputCol("oISBN");
        StringIndexerModel rateLI = new StringIndexer().setInputCol("rate").fit(ratingDataset).setOutputCol("orate");
        Pipeline p1 = new Pipeline().setStages(new PipelineStage[]{userIdLI, ISBNLI, rateLI});
        Dataset<Row> data = p1.fit(ratingDataset).transform(ratingDataset);
        
        System.out.println("heeeeeeeeeeeeeeeeeeeere");
        
        data.show();
        */
        Dataset<Row>[] splits = ratingDataset.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];
        ALS als = new ALS().setMaxIter(10).setRegParam(0.01).setUserCol("userId").setItemCol("ISBN").setRatingCol("rate");
        ALSModel model = als.fit(training);
        model.setColdStartStrategy("drop");
        Dataset<Row> predictions = model.transform(test);
        Dataset<Row> r = model.recommendForAllUsers(10);
        System.out.println(" r show");
      
         r.coalesce(1).write().json("output2.json");
        // evaluation crieteria for model
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rate")
                .setPredictionCol("prediction");
        Double rmse = evaluator.evaluate(predictions);
        System.out.println("Root-mean-square error = " + rmse);
    }
}