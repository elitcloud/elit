Decode with Web API
===================

ELIT provides web API that allows anyone to decode raw text into NLP structures using its built-in models.
The web APIs do not require the `ELIT installation <install.html>`_ and can be used by any programming language that supports HTTP request/response.


---------------
Single Document
---------------

The following codes take an input document and run the NLP pipeline for
tokenization, named entity recognition, part-of-speech tagging , morphological analysis, dependency parsing, and coreference resolution using the default parameters:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import requests

         url = 'https://elit.cloud/api/public/decode/doc'

         doc = 'Jinho Choi is a professor at Emory University in Atlanta, GA. ' \
               'Dr. Choi started the Emory NLP Research Group in 2014. ' \
               'He is the founder of the ELIT project.'

         models = [
             {'model': 'elit-en-tok'},
             {'model': 'elit-en-ner-ontonotes'},
             {'model': 'elit-en-pos-general'},
             {'model': 'elit-en-morph'},
             {'model': 'elit-en-dep-general'},
             {'model': 'elit-en-senti-movie-review'},
             {'model': 'e2e-en-coref-ontonotes'}]

         request = {'input': doc, 'models': models}
         r = requests.post(url, json=request)
         print(r.text)

   .. tab:: Java

      .. code-block:: java

         import org.apache.http.HttpResponse;
         import org.apache.http.client.HttpClient;
         import org.apache.http.client.methods.HttpPost;
         import org.apache.http.entity.ContentType;
         import org.apache.http.entity.StringEntity;
         import org.apache.http.impl.client.HttpClientBuilder;
         import org.apache.http.util.EntityUtils;

         public class ELITWebAPI
         {
             public static void main(String[] args) throws Exception
             {
                 HttpPost post = new HttpPost("https://elit.cloud/api/public/decode/doc");
                 HttpClient client = HttpClientBuilder.create().build();

                 String doc = "Jinho Choi is a professor at Emory University in Atlanta, GA. " +
                              "Dr. Choi started the Emory NLP Research Group in 2014. " +
                              "He is the founder of the ELIT project.";

                 String models = "[" +
                         "{\"model\": \"elit-en-tok\"}," +
                         "{\"model\": \"elit-en-ner\"}," +
                         "{\"model\": \"elit-en-pos\"}," +
                         "{\"model\": \"elit-en-morph\"}," +
                         "{\"model\": \"elit-en-dep\"}," +
                         "{\"model\": \"elit-en-dep\"}," +
                         "{\"model\": \"e2e-en-coref\"}]";

                 String request = "{\"input\": " + doc + ", \"models\": " + models + "}";

                 post.setEntity(new StringEntity(request, ContentType.create("application/json")));
                 HttpResponse response = client.execute(post);
                 System.out.println(EntityUtils.toString(response.getEntity()));
             }
         }

      Add the following dependency to ``pom.xml``.

      .. code-block:: xml

         <dependency>
             <groupId>org.apache.httpcomponents</groupId>
             <artifactId>httpclient</artifactId>
             <version>4.5.6</version>
         </dependency>

   .. tab:: Ruby

      .. code-block:: ruby

         text = 'Jinho Choi is a professor at Emory University in Atlanta, GA. ' \
                'Dr. Choi started the Emory NLP Research Group in 2014. ' \
                'He is the founder of the ELIT project.'

   .. tab:: Node.js

      .. code-block:: javascript

         text = 'Jinho Choi is a professor at Emory University in Atlanta, GA. ' +
                'Dr. Choi started the Emory NLP Research Group in 2014. ' +
                'He is the founder of the ELIT project.'

The following shows the output in the JSON format (see the `output format <../documents/output_format.html>`_ for more details):

.. code-block:: json

   {"sens":
      [{"sen_id": 0,
        "tok": ["Jinho", "Choi", "is", "a", "professor", "at", "Emory", "University", "."],
        "ner": [[0, 2, "PERSON"], [6, 8, "ORG"]],
        "pos": ["NNP", "NNP", "VBZ", "DT", "NN", "IN", "NNP", "NNP", "."],
        "mor": [[("jinho", "NN")], [("choi", "NN")], [("be", "VB"), ("", "I_3PS")],
                [("a", "DT")], [ ("pro+", "P"), ("fess", "VB"), ("+or", "N_ER")],
                [("at", "IN")], [("emory", 'NN')], [("university", "NN")], [(".", "PU")]],
        "dep": [[1, "compound"], [2, "nsbj"], [4, "cop"], [4, "det"], [-1, "root"],
                [7, "case"], [7, "compound"], [4, "ppmod"], [4,"punct"]]},
       {"sen_id": 1,
        "tok": ["He","is","the","director","of","EmoryNLP","in","Atlanta",",","GA","."], ...},
       {"sen_id": 2,
        "tok": ["Dr.","Choi","is","happy","to","be","at","AWS","re:Invent","2018","."], ...}],
     "coref": [{[0, 0, 2], [1, 0, 1], [2, 0, 2]}]}


------------------
Parameter Settings
------------------

Some models can take parameters for further customizations

### Hello

------------------
Multiple Documents
------------------