Decode with Python API
======================

Once ELIT is `installed <install.html>`_, NLP components can be used to decode raw text into linguistic structures.


Create NLP Pipeline
-------------------

The followings show how to create an NLP pipeline with 6 core components:

.. code:: python

   from elit.component import EnglishTokenizer
   from elit.component import EnglishMorphAnalyzer
   from elit.component import POSFlairTagger
   from elit.component import NERFlairTagger
   from elit.component import DEPBiaffineParser
   from elit.component import SDPBiaffineParser

   tok = EnglishTokenizer()
   morph = EnglishMorphAnalyzer()
   pos = POSFlairTagger()
   ner = NERFlairTagger()
   dep = DEPBiaffineParser()
   sdp = SDPBiaffineParser()

   tools = [tok, morph, pos, ner, dep, sdp]

Take a look at the individual tool page for more details about available components:

* `Tokenizer <../tools/tokenization.html>`_
* `Morphological analyser <../tools/morphological_analysis.html>`_
* `Part-Of-Speech tagger <../tools/part_of_speech_tagging.html>`_
* `Named Entity recognizer <../tools/named_entity_recognition.html>`_
* `Dependency parser <../tools/dependency_parsing.html>`_
* `Semantic dependency parser <../tools/semantic_dependency_parsing.html>`_


Import Models
-------------

All `pre-trained models <../documentation/models.html>`_ are publicly available in the `ELIT's S3 bucket <http://elit-models.s3.amazonaws.com>`_.
The followings show how to import models for ``pos``, ``ner``, ``dep``, and ``sdp``:

.. code:: python

   from elit.resources.pre_trained_models import ELIT_POS_FLAIR_EN_MIXED
   from elit.resources.pre_trained_models import ELIT_NER_FLAIR_EN_ONTONOTES
   from elit.resources.pre_trained_models import ELIT_DEP_BIAFFINE_EN_MIXED
   from elit.resources.pre_trained_models import ELIT_SDP_BIAFFINE_EN_MIXED

   pos.load(ELIT_POS_FLAIR_EN_MIXED)
   ner.load(ELIT_NER_FLAIR_EN_ONTONOTES)
   dep.load(ELIT_DEP_BIAFFINE_EN_MIXED)
   sdp.load(ELIT_SDP_BIAFFINE_EN_MIXED)

The ``load`` function takes two parameters, ``model_path`` and ``model_root``:

* ``model_path`` indicates either the name of the model (e.g., ``elit_pos_flair_en_mixed_20190626``) or a public URL to the model file compressed in the zip format (e.g., https://elit-models.s3-us-west-2.amazonaws.com/elit_pos_flair_en_mixed_20190626.zip).
* ``model_root`` indicates the root directory in the local machine where all models are saved, and has the default value of ``~/.elit/models/``.

If ``model_path`` points to an URL, this function downloads the remote file and unzips it under the directory indicated by ``model_root``, which will create a directory with the same model name (e.g., ``~/.elit/models/elit_pos_flair_en_mixed_20190626/``).
Each model directory has the configuration file, ``config.json``, that may indicate dependencies to other models, in which case, it will recursively download all necessary models and uncompresses them under the ``model_root`` directory (see the `Train with CLI <train_cli.html>`_ page for more details about how models are saved).


Prepare Raw Text
----------------

The followings show how to prepare raw text for decoding:

.. code:: python

   docs = [
       'Emory University is a private research university in Atlanta, Georgia. The university is ranked 21st nationally according to U.S. News.',
       'Emory University was founded in 1836 by the Methodist Episcopal Church. It was named in honor of John Emory who was a Methodist bishop.'
   ]

ELIT accepts a list of strings as input, where each string represents a document such that there are two documents in ``docs``.


Decode with NLP Pipeline
------------------------

Finally, the followings show how to decode the raw text with the NLP pipeline:

.. code:: python

   for tool in tools:
       docs = tool.decode(docs)

The ``decode`` function in the `tokenizer <../tools/tokenization.html>`_ takes a list of strings and returns a list of `Document <../documentation/structures.html#document>`_, whereas the ``decode`` functions in other models take a list of document objects and return a list of the same objects where the decoding results are added as distinct fields (see the `NLP Output`_ below).


All Together
------------

The followings put all the codes together:

.. code:: python

   from elit.component import EnglishTokenizer
   from elit.component import EnglishMorphAnalyzer
   from elit.component import POSFlairTagger
   from elit.component import NERFlairTagger
   from elit.component import DEPBiaffineParser
   from elit.component import SDPBiaffineParser

   from elit.resources.pre_trained_models import ELIT_POS_FLAIR_EN_MIXED
   from elit.resources.pre_trained_models import ELIT_NER_FLAIR_EN_ONTONOTES
   from elit.resources.pre_trained_models import ELIT_DEP_BIAFFINE_EN_MIXED
   from elit.resources.pre_trained_models import ELIT_SDP_BIAFFINE_EN_MIXED

   tok = EnglishTokenizer()
   morph = EnglishMorphAnalyzer()
   pos = POSFlairTagger().load(ELIT_POS_FLAIR_EN_MIXED)
   ner = NERFlairTagger().load(ELIT_NER_FLAIR_EN_ONTONOTES)
   dep = DEPBiaffineParser().load(ELIT_DEP_BIAFFINE_EN_MIXED)
   sdp = SDPBiaffineParser().load(ELIT_SDP_BIAFFINE_EN_MIXED)

   tools = [tok, morph, pos, ner, dep, sdp]

   docs = [
       'Emory University is a private research university in Atlanta, Georgia. The university is ranked 21st nationally according to U.S. News.',
       'Emory University was founded in 1836 by the Methodist Episcopal Church. It was named in honor of John Emory who was a Methodist bishop.'
   ]

   for tool in tools:
       docs = tool.decode(docs)

    print(docs)


NLP Output
----------

The followings show the printed output of the above code:

.. code:: json

   To be filled

See the `Formats <../documentation/formats.html>`_ page for more details about how the decoding results are added to `Document <../documentation/structures.html#document>`_.
