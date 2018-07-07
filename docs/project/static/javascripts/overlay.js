$(document).ready(function() {
  $(".overlay").click(function(event) {
    event.preventDefault();
    $('body').css('overflow', 'hidden');
    var url=$(this).attr("href");
    $(".overlay-iframe").attr('src', url);
    $(".overlay-new").attr('href', url);
    if (url.indexOf("notebook") != -1) {
      var nburl="https://github.com/markroxor/fromscratchtoml/tree/master/docs/notebooks" + url.substring(url.lastIndexOf("/"), url.lastIndexOf(".")) + ".ipynb"
      $(".overlay-cloud").attr('href', nburl);
    }
    $(".overlay-bg").fadeIn(500);
  });

  $(document).keyup(function(event) {
    if (event.keyCode == 27) {
      closeOverlay(event);
    }
  });

  $(".overlay-close").click(function(event) {
    closeOverlay(event);
  });
});


function closeOverlay(event) {
  event.preventDefault();
  $(".overlay-bg").fadeOut(500);
  $('body').css('overflow', 'visible');
}
