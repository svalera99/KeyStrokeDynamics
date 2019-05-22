window.onload = function(){
        var tmp_key = {};
        var key_log = [];

        var start_t = null;
        var start_prev_t = null;
        $('#main_input').on("keydown", function(event){
            if(start_t == null){ start_t = $.now(); }
            tmp_key.vk = event.key;

            let end_t = $.now();
            var time_elapsed = end_t - (start_prev_t == null ? end_t : start_prev_t);
            tmp_key.diff_prev_t = time_elapsed;
        }).on("keyup", function(element){
            let end_t = $.now();
            var time_elapsed = end_t - (start_t == null ? end_t : start_t);
            tmp_key.vk = event.key;

            tmp_key.start_t = start_t;
            tmp_key.end_t = end_t;
            tmp_key.diff_t = time_elapsed;
            key_log.push(tmp_key);
            //console.log(tmp_key);

            start_prev_t = start_t;
            start_t = null;
            //console.log(["Key: ", tmp_key.vk, ", time prev: ", tmp_key.diff_prev_t, ", time: ", time_elapsed, "ms"].join(""));
            tmp_key = {};
        });


    $("#main_form").on("keyup",function (val) {
        if (val.which == 13){
                for (let i in key_log){
                    if (!("diff_prev_t" in key_log[i])){
                        key_log[i]["diff_prev_t"] = null;
                    }
                    if (key_log[i]["diff_t"] === 0){
                        key_log[i]["diff_t"] = null;
                    }
                }
            $.ajax({
                "type": "POST",
                "url": "/reg",
                "data": JSON.stringify(key_log),
                contentType: 'application/json;charset=UTF-8'
            }).done(function(data , status){
                console.log("Success: " + data + status);
                document.getElementById("main_form").reset();
            }).fail(function(data , status ){
                console.log("Error: " + data+ status);
                document.getElementById("main_form").reset();
            });
     }

    })
};

