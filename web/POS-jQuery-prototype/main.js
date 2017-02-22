$("#sentence").click(function(){
    console.log("sentence clicked!");
});


var $nouns = $('.noun');

var isHovered = false;

$nouns.hover(function () {
  $(this).toggleClass('active');
  $nouns.toggleClass('hover');
    
    if (!isHovered) {
        $('#POS-label').html("Part of speech: <span id = \"POS\"> noun<span>");
        $('#POS').css("color","dodgerblue");
        isHovered = true;
    } else {
        $('#POS-label').html("Part of speech:");
        isHovered = false;
    }
})

