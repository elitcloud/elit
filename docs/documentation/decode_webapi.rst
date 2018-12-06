Decode with Web API
===================

ELIT provides web API that allows anyone to decode raw text into NLP structures using its `built-in models <models.html>`_.
The web API does not require installation and can be used by any programming language that supports HTTP request/response.


---------------
Single Document
---------------

The following code takes a single document and runs the NLP pipeline for
tokenization, part-of-speech tagging, named entity recognition, dependency parsing, morphological analysis, and coreference resolution using default parameters:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import requests

         url = 'https://elit.cloud/api/public/decode/doc'

         doc = 'Jinho Choi is a professor at Emory University in Atlanta, GA. ' \
               'Dr. Choi started the Emory NLP Research Group in 2014. ' \
               'He is the founder of the ELIT project.'

         models = [
             {'model': 'elit_tok_lexrule_en'},
             {'model': 'elit_pos_rnn_en_mixed'},
             {'model': 'elit_ner_cnn_en_ontonotes'},
             {'model': 'elit_dep_biaffine_en_mixed'},
             {'model': 'elit_morph_lexrule_en'},
             {'model': 'uw_coref_e2e_en_ontonotes'}]

         request = {'input': doc, 'models': models}
         r = requests.post(url, json=request)
         print(r.text)

   .. tab:: Java

      .. code-block:: java

         import org.apache.http.HttpEntity;
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

             String text = "Jinho Choi is a professor at Emory University in Atlanta, GA. " +
                           "Dr. Choi started the Emory NLP Research Group in 2014. " +
                           "He is the founder of the ELIT project.";

             String models = "[" +
                     "{\"model\":\"elit_tok_lexrule_en\"}, " +
                     "{\"model\":\"elit_pos_rnn_en_mixed\"}, " +
                     "{\"model\":\"elit_ner_cnn_en_ontonotes\"}, " +
                     "{\"model\":\"elit_dep_biaffine_en_mixed\"}, " +
                     "{\"model\":\"elit_morph_lexrule_en\"}, " +
                     "{\"model\":\"uw_coref_e2e_en_ontonotes\"}]";

             String request = "{\"input\":\"" + text + "\",\"models\":" + models + "}";
             post.setEntity(new StringEntity(request, ContentType.create("application/json")));
             HttpResponse response = client.execute(post);
             HttpEntity e = response.getEntity();
             System.out.println(EntityUtils.toString(e));
           }
         }

      Add the following dependency to ``pom.xml``.

      .. code-block:: xml

         <dependency>
             <groupId>org.apache.httpcomponents</groupId>
             <artifactId>httpclient</artifactId>
             <version>4.5.6</version>
         </dependency>

The following shows the output in the JSON format (see the `output format <../documentation/output_format.html>`_ for more details):

.. code-block:: javascript

   {"sens": [
       {"sid": 0,
        "tok": ["Jinho", "Choi", "is", "a", "professor", "at", "Emory", "University", "in", "Atlanta", ", ", "GA", "."],
        "off": [[0, 5], [6, 10], [11, 13], [14, 15], [16, 25], [26, 28], [29, 34], [35, 45], [46, 48], [49, 56], [56, 57], [58, 60], [60, 61]],
        "pos": ["NNP", "NNP", "VBZ", "DT", "NN", "IN", "NNP", "NNP", "IN", "NNP", ", ", "NNP", "."],
        "ner": [[0, 2, "PERSON"], [6, 8, "ORG"], [9, 10, "GPE"], [11, 12, "GPE"]],
        "dep": [[1, "com"], [4, "nsbj"], [4, "cop"], [4, "det"], [-1, "root"], [7, "case"],
                [7, "com"], [4, "ppmod"], [9, "case"], [7, "ppmod"], [9, "p"], [9, "appo"], [4, "p"]],
        "morph": [[["jinho", "NN"]], [["choi", "NN"]], [["be", "VB"], ["", "I_3PS"]],
                  [["a", "DT"]], [["profess", "VB"], ["+or", "N_ER"]],
                  [["at", "IN"]], [["emory", "NN"]], [["university", "NN"]], [["in", "IN"]],
                  [["atlanta", "NN"]], [[", ", "PU"]], [["ga", "NN"]], [[".", "PU"]]]},
       {"sid": 1,
        "tok": ["Dr.", "Choi", "started", "the", "Emory", "NLP", "Research", "Group", "in", "2014", "."],
        "off": [[62, 65], [66, 70], [71, 78], [79, 82], [83, 88], [89, 92], [93, 101], [102, 107], [108, 110], [111, 115], [115, 116]],
        "pos": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "NNP", "IN", "CD", "."],
        "ner": [[0, 2, "PERSON"], [3, 8, "ORG"], [9, 10, "DATE"]],
        "dep": [[1, "com"], [2, "nsbj"], [-1, "root"], [7, "det"], [7, "com"], [7, "com"],
                [7, "com"], [2, "obj"], [9, "case"], [2, "ppmod"], [2, "p"]],
        "morph": [[["dr.", "NN"]], [["choi", "NN"]], [["start", "VB"], ["+ed", "I_PST"]],
                  [["the", "DT"]], [["emory", "NN"]], [["nlp", "NN"]], [["research", "NN"]],
                  [["group", "NN"]], [["in", "IN"]], [["2014", "CD"]], [[".", "PU"]]]},
       {"sid": 2,
        "tok": ["He", "is", "the", "founder", "of", "the", "ELIT", "project", "."],
        "off": [[117, 119], [120, 122], [123, 126], [127, 134], [135, 137], [138, 141], [142, 146], [147, 154], [154, 155]],
        "ner": [[6, 7, "ORG"]],
        "dep": [[3, "nsbj"], [3, "cop"], [3, "det"], [-1, "root"], [7, "case"], [7, "det"],
                [7, "com"], [3, "ppmod"], [2, "p"]],
        "morph": [[["he", "PR"]], [["be", "VB"], ["", "I_3PS"]], [["the", "DT"]],
                  [["found", "VB"], ["+er", "N_ER"]], [["of", "IN"]], [["the", "DT"]],
                  [["elit", "NN"]], [["project", "NN"]], [[".", "PU"]]]}]
    "coref": [
       [[0, 0, 2], [1, 0, 2], [2, 0, 1]]]}

See the `available models <models.html>`_ page for the list of all built-in models and their parameter settings.


------------------
Multiple Documents
------------------

`Coming soon`.