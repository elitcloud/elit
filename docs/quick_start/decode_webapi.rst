Decode with Web APIs
====================

ELIT provides web APIs to decode raw text into NLP structures using `pre-trained models <../documentation/models.html>`_.
The web APIs do not require `installation <install.html>`_ and can be used by any programming language that supports HTTP request/response.


Decode via HTTP
---------------

The followings show how to send a list of documents to ELIT and receive the `NLP output`_ consisting of decoding results from 6 models:

* Tokenization: `elit_tok_lexrule_en <../tools/tokenization.html#english-tokenizer>`_
* Morphological analysis: `elit_morph_lexrule_en <../tools/morphological_analysis.html#english-analyzer>`_
* Part-Of-Speech tagging: `elit_pos_flair_en_mixed <../tools/part_of_speech_tagging.html#flair-tagger>`_
* Named Entity recognization: `elit_ner_flair_en_ontonotes <../tools/named_entity_recognition.html#flair-tagger>`_
* Dependency parsing: `elit_dep_biaffine_en_mixed <../tools/dependency_parsing.html#biaffine-parser>`_
* Semantic dependency parsing: `elit_sdp_biaffine_en_mixed <../tools/semantic_dependency_parsing.html#biaffine-parser>`_

Take a look at the individual model page for more details and their parameter settings.

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import requests

         url = 'https://elit.cloud/api/public/decode/raw'

         docs = [
             'Emory University is a private research university in Atlanta, Georgia. The university is ranked 21st nationally according to U.S. News.',
             'Emory University was founded in 1836 by the Methodist Episcopal Church. It was named in honor of John Emory who was a Methodist bishop.']

         models = [
             {'model': 'elit_tok_lexrule_en'},
             {'model': 'elit_pos_flair_en_mixed'},
             {'model': 'elit_morph_lexrule_en'},
             {'model': 'elit_ner_flair_en_ontonotes'},
             {'model': 'elit_dep_biaffine_en_mixed'},
             {'model': 'elit_sdp_biaffine_en_mixed'}]

         request = {'input': docs, 'models': models}
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
                 HttpPost post = new HttpPost("https://elit.cloud/api/public/decode/raw");
                 HttpClient client = HttpClientBuilder.create().build();

                 String docs = String.format("["\%s\", "\%s\"]",
                     "Emory University is a private research university in Atlanta, Georgia. The university is ranked 21st nationally according to U.S. News.",
                     "Emory University was founded in 1836 by the Methodist Episcopal Church. It was named in honor of John Emory who was a Methodist bishop.");

                 String models = String.format("[{\"model\": \"%s\"}, {\"model\": \"%s\"}, {\"model\": \"%s\"}, {\"model\": \"%s\"}, {\"model\": \"%s\"}, {\"model\": \"%s\"}]",
                     "elit_tok_lexrule_en",
                     "elit_pos_flair_en_mixed",
                     "elit_morph_lexrule_en",
                     "elit_ner_flair_en_ontonotes",
                     "elit_dep_biaffine_en_mixed",
                     "elit_sdp_biaffine_en_mixed");

                 String request = "{\"input\": " + docs + ", \"models\": " + models + "}";

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
             <version>4.5.9</version>
         </dependency>


NLP Output
----------

The followings show the printed output of the above code:

.. code:: python

   To be filled

See the `Formats <../documentation/formats.html>`_ page for more details about how the decoding results are added to `Document <../documentation/structures.html#document>`_.
